---
tags: [Машинное обучение]
---

# Mixture of Experts (MoE)

Это одна из самых мощных идей в современном машинном обучении для создания огромных, но эффективных моделей. Давайте разберем ее подробно.

## Основная концепция

Представьте, что у вас есть сложная задача, например, перевод текста с любого языка на любой. Вместо того чтобы пытаться создать одного гигантского "универсального гения", который знает всё, вы нанимаете команду экспертов (эксперт по немецкому, эксперт по японскому, эксперт по юридическим текстам, эксперт по разговорной речи и т.д.).

Задача "роутера" — посмотреть на incoming текст и решить: "Окей, это японский технический мануал. Дайте мне эксперта по японскому и эксперта по технике. Остальные могут отдыхать."

В чем выигрыш? Мы получаем качество огромной модели (много экспертов), но вычислительная стоимость за один forward pass примерно равна стоимости нескольких экспертов, а не всех сразу.

## Ключевые компоненты архитектуры MoE

1. Эксперты (Experts): Это несколько независимых нейронных сетей (обычно с одинаковой архитектурой, но разными весами). Чаще всего это Feed-Forward Networks (FFN) внутри трансформера.
2. Роутер (Router или Gating Network): Небольшая сеть (часто просто линейный слой + функция активации softmax), которая принимает входные данные и возвращает веса для каждого эксперта. Она решает, насколько каждый эксперт важен для текущего входного примера.
3. Механизм разреженность (Sparsity): Вместо того чтобы активировать всех экспертов, мы выбираем только Top-K (обычно K=1 или 2) экспертов с наибольшими весами от роутера. Это основа эффективности.

## Пример кода на Python с использованием PyTorch

Упрощенный слой MoE для задачи классификации.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """Один эксперт — простой FFN"""
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class MoELayer(nn.Module):
    """Слой Mixture of Experts"""
    def __init__(self, input_size, output_size, hidden_size, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Создаем пул экспертов
        self.experts = nn.ModuleList(
            [Expert(input_size, output_size, hidden_size) for _ in range(num_experts)]
        )

        # Роутер (шлюз)
        self.router = nn.Linear(input_size, num_experts)

        # Вспомогательный слой для балансировки нагрузки (об этом ниже)
        self.aux_loss = 0.0

    def forward(self, x):
        # x shape: [batch_size, input_size]
        batch_size = x.shape[0]

        # 1. Получаем логиты от роутера
        router_logits = self.router(x) # [batch_size, num_experts]

        # 2. Применяем softmax для получения вероятностей (весов экспертов)
        router_probs = F.softmax(router_logits, dim=-1) # [batch_size, num_experts]

        # 3. Выбираем Top-K экспертов для каждого элемента в батче
        top_k_weights, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        # top_k_weights: [batch_size, top_k], top_k_indices: [batch_size, top_k]

        # 4. Нормализуем веса Top-K экспертов для каждого примера
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        # 5. Инициализируем итоговый выходной тензор
        final_output = torch.zeros_like(x) # [batch_size, output_size]

        # 6. Реализуем механизм Sparse Dispatching
        # Мы будем накапливать (accumulate) выходы каждого эксперта, взвешенные на top_k_weights.
        for i in range(self.top_k):
            # Для каждого индекса в Top-K (например, сначала для 1-го, потом для 2-го)
            expert_mask = top_k_indices == i
            # Находим для каких батчей i-й топовый эксперт - это j-й реальный эксперт
            for expert_id in range(self.num_experts):
                # Создаем маску: [batch_size,], где True для примеров, которым нужен данный эксперт
                batch_indices, _ = torch.where(expert_mask[:, i] == expert_id)
                if len(batch_indices) == 0:
                    continue # Этот эксперт не нужен ни одному примеру в этой позиции Top-K

                # Выбираем данные, которые пойдут этому эксперту
                expert_input = x[batch_indices]
                # Пропускаем их через выбранного эксперта
                expert_output = self.experts[expert_id](expert_input)
                # Получаем веса для этих данных и этого эксперта
                weight = top_k_weights[batch_indices, i].unsqueeze(1)
                # Взвешиваем выход эксперта и добавляем к финальному выходу
                final_output[batch_indices] += weight * expert_output

        # 7. Добавляем Auxiliary Loss для балансировки нагрузки (CRUCIAL!)
        self.compute_auxiliary_loss(router_probs, top_k_indices)

        return final_output

    def compute_auxiliary_loss(self, router_probs, top_k_indices):
        """
        Auxiliary Loss для балансировки нагрузки.
        Поощряет равномерное распределение нагрузки между экспертами.
        Подсчитывает, какую долю батча обработал каждый эксперт.
        """
        # Считаем, сколько раз каждый эксперт был в Top-K
        expert_usage = torch.zeros(self.num_experts, device=router_probs.device)
        # Для каждого эксперта, подсчитываем количество раз, когда он был выбран
        for expert_id in range(self.num_experts):
            expert_usage[expert_id] = (top_k_indices == expert_id).float().sum()

        # Доля нагрузки на каждого эксперта от общего числа выборок
        load_per_expert = expert_usage / torch.sum(expert_usage)

        # Коэффициент важности auxiliary loss
        aux_loss_coef = 0.01
        # Рассчитываем потерю: дисперсия нагрузки (чем равномернее, тем лучше)
        self.aux_loss = aux_loss_coef * torch.std(load_per_expert) * self.num_experts

# Пример использования
input_size = 100
output_size = 100
hidden_size = 64
num_experts = 8
top_k = 2

model = MoELayer(input_size, output_size, hidden_size, num_experts, top_k)

# Сгенерируем случайные данные
x = torch.randn(32, input_size) # batch_size=32

# Forward pass
output = model(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Auxiliary Loss: {model.aux_loss.item()}")
```

## 4. Критически важный момент: Балансировка нагрузки (Load Balancing)

Самая большая проблема MoE — "сильный становится сильнее". Роутер может быстро понять, что 1-2 эксперта хороши, и начать использовать только их, оставляя других необученными (проблема "мёртвых экспертов").

Решение: Auxiliary Loss
Код выше включает compute_auxiliary_loss. Эта потеря:

1. Считает, насколько равномерно нагрузка распределена между экспертами.
2. Добавляется к главной функции потерь задачи (main_loss + aux_loss).
3. Наказывает модель, если эксперты используются неравномерно, заставляя роутер "распределять работу" более справедливо.

### Вывод:

MoE — это блестящая архитектура, которая использует идею "нейроморфного роутинга" на уровне software. Она позволяет на порядки увеличить количество параметров модели без пропорционального роста вычислений, делегируя решение о том, какую часть сети или модель использовать.
