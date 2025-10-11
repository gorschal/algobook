---
tags:
  [
    Игровая разработка,
    Экономика и финансы,
    Биотехнологии и медицина,
    Машинное обучение,
  ]
---

# Цепи Маркова: подробное объяснение с примерами на Python

## Что такое цепь Маркова?

**Цепь Маркова** - это математическая модель, описывающая последовательность событий, где вероятность каждого события зависит только от предыдущего состояния (свойство **отсутствия памяти**).

## Основные понятия

### 1. Состояния

$ S = {s₁, s₂, ..., sₙ} $ - множество возможных состояний системы

### 2. Матрица переходных вероятностей

$ P = [pᵢⱼ] - матрица, где pᵢⱼ = P(Xₙ₊₁ = j | Xₙ = i) $

- Сумма вероятностей в каждой строке = 1

### 3. Свойство Маркова

$ P(Xₙ₊₁ = j | Xₙ = i, Xₙ₋₁ = iₙ₋₁, ..., X₀ = i₀) = P(Xₙ₊₁ = j | Xₙ = i) $

## Типы цепей Маркова

### 1. Однородные vs Неоднородные

- **Однородные**: вероятности переходов не зависят от времени
- **Неоднородные**: вероятности меняются со временем

### 2. Эргодические vs Поглощающие

- **Эргодические**: из любого состояния можно достичь любого другого
- **Поглощающие**: есть состояния, из которых нельзя выйти

## Примеры реализации

### Пример 1: Простая цепь Маркова для погоды

```python
import numpy as np
import matplotlib.pyplot as plt

class MarkovChain:
    def __init__(self, transition_matrix, states):
        """
        transition_matrix: матрица переходных вероятностей
        states: список состояний
        """
        self.transition_matrix = np.array(transition_matrix)
        self.states = states
        self.state_index = {state: i for i, state in enumerate(states)}
        self.current_state = None

    def set_state(self, state):
        """Установить текущее состояние"""
        self.current_state = self.state_index[state]

    def next_state(self):
        """Перейти к следующему состоянию"""
        if self.current_state is None:
            raise ValueError("State not set")

        # Выбираем следующее состояние на основе вероятностей перехода
        next_state_idx = np.random.choice(
            len(self.states),
            p=self.transition_matrix[self.current_state]
        )
        self.current_state = next_state_idx
        return self.states[next_state_idx]

    def simulate(self, start_state, n_steps):
        """Симуляция цепи на n_steps шагов"""
        self.set_state(start_state)
        sequence = [start_state]

        for _ in range(n_steps):
            sequence.append(self.next_state())

        return sequence

    def stationary_distribution(self, max_iter=1000, tol=1e-6):
        """Нахождение стационарного распределения"""
        n = len(self.states)
        pi = np.ones(n) / n  # начальное распределение

        for _ in range(max_iter):
            pi_new = pi @ self.transition_matrix
            if np.linalg.norm(pi_new - pi) < tol:
                break
            pi = pi_new

        return {state: pi[i] for i, state in enumerate(self.states)}

# Пример: Модель погоды
# Состояния: Sunny, Cloudy, Rainy
weather_matrix = [
    [0.6, 0.3, 0.1],  # Sunny -> [Sunny, Cloudy, Rainy]
    [0.4, 0.4, 0.2],  # Cloudy -> [Sunny, Cloudy, Rainy]
    [0.2, 0.3, 0.5]   # Rainy -> [Sunny, Cloudy, Rainy]
]

weather_states = ['Sunny', 'Cloudy', 'Rainy']

# Создаем цепь Маркова
weather_chain = MarkovChain(weather_matrix, weather_states)

# Симуляция на 20 дней
weather_sequence = weather_chain.simulate('Sunny', 20)
print("Последовательность погоды:", weather_sequence)

# Стационарное распределение
stationary = weather_chain.stationary_distribution()
print("\nСтационарное распределение:")
for state, prob in stationary.items():
    print(f"{state}: {prob:.3f}")
```

### Пример 2: Генератор текста на основе цепей Маркова

```python
import re
from collections import defaultdict
import random

class TextMarkovGenerator:
    def __init__(self, order=2):
        self.order = order  # порядок цепи
        self.model = defaultdict(lambda: defaultdict(int))
        self.starts = []  # начальные состояния

    def train(self, text):
        """Обучение модели на тексте"""
        # Очистка текста и разбиение на слова
        words = re.findall(r'\b\w+\b', text.lower())

        if len(words) <= self.order:
            return

        # Создание модели
        for i in range(len(words) - self.order):
            # Текущее состояние (n слов)
            state = tuple(words[i:i + self.order])

            # Следующее слово
            next_word = words[i + self.order]

            # Запоминаем начальные состояния
            if i == 0:
                self.starts.append(state)

            # Обновляем счетчики переходов
            self.model[state][next_word] += 1

    def _normalize_probabilities(self):
        """Нормализация вероятностей"""
        for state in self.model:
            total = sum(self.model[state].values())
            for word in self.model[state]:
                self.model[state][word] /= total

    def generate_text(self, length=50):
        """Генерация текста"""
        if not self.model:
            return "Модель не обучена"

        self._normalize_probabilities()

        # Выбираем случайное начальное состояние
        current_state = random.choice(self.starts)
        result = list(current_state)

        for _ in range(length - self.order):
            if current_state not in self.model:
                break

            # Выбираем следующее слово на основе вероятностей
            next_words = list(self.model[current_state].keys())
            probabilities = list(self.model[current_state].values())

            next_word = random.choices(next_words, weights=probabilities)[0]
            result.append(next_word)

            # Обновляем текущее состояние
            current_state = tuple(result[-self.order:])

        return ' '.join(result)

# Пример использования
text = """
Цепи Маркова это математические модели которые используются во многих областях
науки и техники Они особенно полезны для анализа случайных процессов и
прогнозирования будущих состояний системы на основе текущего состояния
Модели Маркова применяются в лингвистике экономике физике и компьютерных науках
"""

generator = TextMarkovGenerator(order=2)
generator.train(text)

print("Сгенерированный текст:")
for i in range(3):
    print(f"{i+1}. {generator.generate_text(10)}")
```

### Пример 3: Анализ свойств цепи Маркова

```python
def analyze_markov_chain(transition_matrix, states):
    """Анализ свойств цепи Маркова"""
    P = np.array(transition_matrix)
    n = len(states)

    print("Анализ цепи Маркова:")
    print("=" * 50)

    # Проверка стохастичности матрицы
    row_sums = P.sum(axis=1)
    is_stochastic = np.allclose(row_sums, np.ones(n))
    print(f"Матрица стохастическая: {is_stochastic}")

    # Проверка эргодичности
    # Цепь эргодическая, если она неприводимая и апериодическая

    # Вычисление стационарного распределения
    # Решаем систему: πP = π, ∑π = 1
    A = np.vstack([(P.T - np.eye(n))[:-1], np.ones(n)])
    b = np.hstack([np.zeros(n-1), 1])

    try:
        stationary = np.linalg.solve(A.T @ A, A.T @ b)
        print("\nСтационарное распределение:")
        for i, state in enumerate(states):
            print(f"  {state}: {stationary[i]:.4f}")
    except:
        print("Не удалось найти стационарное распределение")

    # Анализ коммуникативных классов
    print(f"\nКоличество состояний: {n}")

    return stationary

# Анализ нашей цепи погоды
analyze_markov_chain(weather_matrix, weather_states)
```

### Пример 4: Визуализация цепи Маркова

```python
import networkx as nx

def visualize_markov_chain(transition_matrix, states):
    """Визуализация цепи Маркова как графа"""
    G = nx.DiGraph()

    # Добавляем узлы
    for state in states:
        G.add_node(state)

    # Добавляем ребра с весами
    for i, state_from in enumerate(states):
        for j, state_to in enumerate(states):
            prob = transition_matrix[i][j]
            if prob > 0:
                G.add_edge(state_from, state_to, weight=prob, label=f"{prob:.2f}")

    # Рисуем граф
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)

    # Рисуем узлы
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue')

    # Рисуем подписи узлов
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

    # Рисуем ребра
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]

    nx.draw_networkx_edges(G, pos, edge_color='gray',
                          arrows=True, arrowsize=20,
                          width=[w * 5 for w in weights])

    # Подписи ребер (вероятности)
    edge_labels = {(u, v): f"{d['weight']:.2f}"
                   for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title("Визуализация цепи Маркова", size=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Визуализируем цепь погоды
visualize_markov_chain(weather_matrix, weather_states)
```

### Пример 5: Моделирование траекторий и анализ сходимости

```python
def analyze_convergence(chain, start_state, n_simulations=1000, max_steps=100):
    """Анализ сходимости к стационарному распределению"""
    state_frequencies = {state: [] for state in chain.states}

    for step in range(1, max_steps + 1):
        # Множественное моделирование
        final_states = []
        for _ in range(n_simulations):
            sequence = chain.simulate(start_state, step)
            final_states.append(sequence[-1])

        # Вычисление частот
        for state in chain.states:
            freq = final_states.count(state) / n_simulations
            state_frequencies[state].append(freq)

    # Визуализация сходимости
    plt.figure(figsize=(12, 8))

    for state in chain.states:
        plt.plot(range(1, max_steps + 1), state_frequencies[state],
                label=state, linewidth=2)

    # Стационарное распределение
    stationary = chain.stationary_distribution()
    for state, prob in stationary.items():
        plt.axhline(y=prob, color='gray', linestyle='--', alpha=0.7)
        plt.text(max_steps, prob, f' {state}: {prob:.3f}', va='center')

    plt.xlabel('Количество шагов')
    plt.ylabel('Вероятность')
    plt.title('Сходимость к стационарному распределению')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Анализ сходимости для цепи погоды
analyze_convergence(weather_chain, 'Sunny', n_simulations=500, max_steps=50)
```
