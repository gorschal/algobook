---
tags: [Экономика и финансы]
---

# Автоматический маркет-мейкинг (AMM)

**Автоматический маркет-мейкинг (AMM)** - это алгоритмический протокол, который автоматически обеспечивает ликвидность на децентрализованных биржах (DEX) без традиционных маркет-мейкеров. Вместо ордербуков `AMM` используют математические формулы для определения цены и выполнения сделок.

## Основные концепции

### 1. Константный продукт (Constant Product Formula)

Наиболее популярная модель, используемая в Uniswap:

```
x * y = k
```

где:

- `x` - количество токена A в пуле
- `y` - количество токена B в пуле
- `k` - константа продукта

### 2. Ценообразование

Цена определяется соотношением резервов в пуле:

```
Price of A in terms of B = y / x
Price of B in terms of A = x / y
```

## Реализация базового AMM на Python

```python
import math
from decimal import Decimal

class ConstantProductAMM:
    def __init__(self, token_a: str, token_b: str, initial_a: float, initial_b: float):
        self.token_a = token_a
        self.token_b = token_b
        self.reserve_a = Decimal(str(initial_a))
        self.reserve_b = Decimal(str(initial_b))
        self.k = self.reserve_a * self.reserve_b

    def get_price(self, from_token: str, to_token: str) -> Decimal:
        """Получить текущую цену пары"""
        if from_token == self.token_a and to_token == self.token_b:
            return self.reserve_b / self.reserve_a
        elif from_token == self.token_b and to_token == self.token_a:
            return self.reserve_a / self.reserve_b
        else:
            raise ValueError("Invalid token pair")

    def calculate_output_amount(self, input_token: str, input_amount: float) -> Decimal:
        """Рассчитать количество выходного токена при обмене"""
        input_amount_dec = Decimal(str(input_amount))

        if input_token == self.token_a:
            new_reserve_a = self.reserve_a + input_amount_dec
            new_reserve_b = self.k / new_reserve_a
            output_amount = self.reserve_b - new_reserve_b
        elif input_token == self.token_b:
            new_reserve_b = self.reserve_b + input_amount_dec
            new_reserve_a = self.k / new_reserve_b
            output_amount = self.reserve_a - new_reserve_a
        else:
            raise ValueError("Invalid input token")

        # Проверка на достаточность ликвидности
        if output_amount <= 0:
            raise ValueError("Insufficient liquidity")

        return output_amount

    def swap(self, input_token: str, input_amount: float) -> Decimal:
        """Выполнить обмен токенов"""
        output_amount = self.calculate_output_amount(input_token, input_amount)
        input_amount_dec = Decimal(str(input_amount))

        # Обновление резервов
        if input_token == self.token_a:
            self.reserve_a += input_amount_dec
            self.reserve_b -= output_amount
        else:
            self.reserve_b += input_amount_dec
            self.reserve_a -= output_amount

        # k должен оставаться постоянным (с учетом комиссии)
        # В реальных AMM здесь учитывается комиссия
        self.k = self.reserve_a * self.reserve_b

        return output_amount

    def add_liquidity(self, amount_a: float, amount_b: float) -> Decimal:
        """Добавить ликвидность в пул"""
        amount_a_dec = Decimal(str(amount_a))
        amount_b_dec = Decimal(str(amount_b))

        # Проверка соотношения (в реальных AMM это более сложно)
        current_ratio = self.reserve_b / self.reserve_a
        provided_ratio = amount_b_dec / amount_a_dec

        if abs(current_ratio - provided_ratio) > Decimal('0.01'):  # Допуск 1%
            raise ValueError("Invalid ratio provided")

        # Вычисление LP токенов (упрощенно)
        total_liquidity = math.sqrt(self.reserve_a * self.reserve_b)
        liquidity_minted = (amount_a_dec / self.reserve_a) * total_liquidity

        # Обновление резервов
        self.reserve_a += amount_a_dec
        self.reserve_b += amount_b_dec
        self.k = self.reserve_a * self.reserve_b

        return liquidity_minted

    def get_reserves(self) -> tuple:
        """Получить текущие резервы"""
        return float(self.reserve_a), float(self.reserve_b)

# Пример использования
def demonstrate_amm():
    # Создаем AMM пул с 1000 ETH и 2000000 USDT (цена 1 ETH = 2000 USDT)
    amm = ConstantProductAMM("ETH", "USDT", 1000, 2000000)

    print("=== Инициализация пула ===")
    print(f"Резервы: {amm.get_reserves()}")
    print(f"Цена ETH в USDT: {amm.get_price('ETH', 'USDT'):.2f}")
    print(f"Цена USDT в ETH: {amm.get_price('USDT', 'ETH'):.6f}")
    print()

    # Покупка ETH за USDT
    print("=== Покупка 1 ETH за USDT ===")
    usdt_amount = 2000  # Ожидаем потратить ~2000 USDT
    try:
        eth_received = amm.swap("USDT", usdt_amount)
        print(f"Получено ETH: {eth_received:.6f}")
        print(f"Потрачено USDT: {usdt_amount}")
        print(f"Новые резервы: {amm.get_reserves()}")
        print(f"Новая цена ETH: {amm.get_price('ETH', 'USDT'):.2f}")
    except Exception as e:
        print(f"Ошибка: {e}")
    print()

    # Продажа ETH за USDT
    print("=== Продажа 0.5 ETH за USDT ===")
    eth_amount = 0.5
    try:
        usdt_received = amm.swap("ETH", eth_amount)
        print(f"Получено USDT: {usdt_received:.2f}")
        print(f"Потрачено ETH: {eth_amount}")
        print(f"Новые резервы: {amm.get_reserves()}")
        print(f"Новая цена ETH: {amm.get_price('ETH', 'USDT'):.2f}")
    except Exception as e:
        print(f"Ошибка: {e}")
    print()

    # Добавление ликвидности
    print("=== Добавление ликвидности ===")
    try:
        lp_tokens = amm.add_liquidity(10, 20000)  # 10 ETH и 20000 USDT
        print(f"Выпущено LP токенов: {lp_tokens:.2f}")
        print(f"Новые резервы: {amm.get_reserves()}")
    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    demonstrate_amm()
```

## Расширенная реализация с комиссиями

```python
class AdvancedAMM(ConstantProductAMM):
    def __init__(self, token_a: str, token_b: str, initial_a: float, initial_b: float, fee: float = 0.003):
        super().__init__(token_a, token_b, initial_a, initial_b)
        self.fee = Decimal(str(fee))
        self.lp_total_supply = Decimal('0')
        self.lp_balances = {}

    def add_liquidity(self, amount_a: float, amount_b: float, provider: str) -> Decimal:
        """Добавить ликвидность с выпуском LP токенов"""
        amount_a_dec = Decimal(str(amount_a))
        amount_b_dec = Decimal(str(amount_b))

        if self.lp_total_supply == 0:
            # Первое пополнение ликвидности
            liquidity = math.sqrt(amount_a_dec * amount_b_dec)
        else:
            # Пропорциональное пополнение
            liquidity_a = (amount_a_dec / self.reserve_a) * self.lp_total_supply
            liquidity_b = (amount_b_dec / self.reserve_b) * self.lp_total_supply
            liquidity = min(liquidity_a, liquidity_b)

        if liquidity <= 0:
            raise ValueError("Invalid liquidity amount")

        # Обновление резервов
        self.reserve_a += amount_a_dec
        self.reserve_b += amount_b_dec
        self.k = self.reserve_a * self.reserve_b

        # Выпуск LP токенов
        self.lp_total_supply += liquidity
        if provider in self.lp_balances:
            self.lp_balances[provider] += liquidity
        else:
            self.lp_balances[provider] = liquidity

        return liquidity

    def remove_liquidity(self, liquidity: float, provider: str) -> tuple:
        """Удалить ликвидность и получить обратно токены"""
        liquidity_dec = Decimal(str(liquidity))

        if provider not in self.lp_balances or self.lp_balances[provider] < liquidity_dec:
            raise ValueError("Insufficient LP tokens")

        # Вычисление доли
        share = liquidity_dec / self.lp_total_supply

        # Количество возвращаемых токенов
        amount_a = share * self.reserve_a
        amount_b = share * self.reserve_b

        # Обновление резервов и LP токенов
        self.reserve_a -= amount_a
        self.reserve_b -= amount_b
        self.k = self.reserve_a * self.reserve_b
        self.lp_balances[provider] -= liquidity_dec
        self.lp_total_supply -= liquidity_dec

        return float(amount_a), float(amount_b)

    def calculate_output_amount_with_fee(self, input_token: str, input_amount: float) -> Decimal:
        """Рассчитать выходное количество с учетом комиссии"""
        input_amount_dec = Decimal(str(input_amount))
        input_amount_after_fee = input_amount_dec * (Decimal('1') - self.fee)

        if input_token == self.token_a:
            new_reserve_a = self.reserve_a + input_amount_after_fee
            new_reserve_b = self.k / new_reserve_a
            output_amount = self.reserve_b - new_reserve_b
        else:
            new_reserve_b = self.reserve_b + input_amount_after_fee
            new_reserve_a = self.k / new_reserve_b
            output_amount = self.reserve_a - new_reserve_a

        return output_amount

# Демонстрация расширенного AMM
def demonstrate_advanced_amm():
    amm = AdvancedAMM("ETH", "USDT", 100, 200000, 0.003)  # 0.3% комиссия

    print("=== Расширенный AMM с комиссиями ===")

    # Добавляем ликвидность
    lp_tokens = amm.add_liquidity(10, 20000, "provider1")
    print(f"Добавлена ликвидность, выпущено LP токенов: {lp_tokens:.2f}")

    # Обмен с комиссией
    print("\n=== Обмен с комиссией ===")
    usdt_amount = 1000
    eth_received = amm.calculate_output_amount_with_fee("USDT", usdt_amount)
    print(f"При обмене {usdt_amount} USDT получим {eth_received:.6f} ETH")

    # Выполняем обмен
    actual_eth = amm.swap("USDT", usdt_amount)
    print(f"Фактически получено: {actual_eth:.6f} ETH")
    print(f"Новые резервы: {amm.get_reserves()}")

# demonstrate_advanced_amm()
```

## Кривая ликвидности и проскальзывание

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_liquidity_curve():
    """Визуализация кривой ликвидности"""
    # Параметры пула
    k = 1000000  # константа
    x = np.linspace(100, 1000, 100)
    y = k / x

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2, label='Кривая ликвидности x*y=k')
    plt.xlabel('Количество токена A')
    plt.ylabel('Количество токена B')
    plt.title('Кривая ликвидности AMM')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Показываем точку текущего баланса
    current_x, current_y = 500, 2000
    plt.plot(current_x, current_y, 'ro', markersize=8, label='Текущий баланс')
    plt.annotate(f'({current_x}, {current_y})',
                (current_x, current_y),
                textcoords="offset points",
                xytext=(0,10),
                ha='center')

    plt.show()

def calculate_slippage(amm, input_token: str, input_amount: float):
    """Расчет проскальзывания для сделки"""
    spot_price = amm.get_price(input_token, amm.token_b if input_token == amm.token_a else amm.token_a)

    if input_token == amm.token_a:
        output_amount = amm.calculate_output_amount(input_token, input_amount)
        effective_price = Decimal(str(input_amount)) / output_amount
    else:
        output_amount = amm.calculate_output_amount(input_token, input_amount)
        effective_price = output_amount / Decimal(str(input_amount))

    slippage = (effective_price - spot_price) / spot_price * 100

    return {
        'spot_price': float(spot_price),
        'effective_price': float(effective_price),
        'slippage_percent': float(slippage),
        'output_amount': float(output_amount)
    }

# Пример анализа проскальзывания
def slippage_analysis():
    amm = ConstantProductAMM("ETH", "USDT", 1000, 2000000)

    print("=== Анализ проскальзывания ===")

    trade_sizes = [100, 1000, 10000, 50000]  # USDT

    for size in trade_sizes:
        result = calculate_slippage(amm, "USDT", size)
        print(f"\nРазмер сделки: {size} USDT")
        print(f"Спот цена: {result['spot_price']:.2f}")
        print(f"Эффективная цена: {result['effective_price']:.2f}")
        print(f"Проскальзывание: {result['slippage_percent']:.4f}%")
        print(f"Получено ETH: {result['output_amount']:.6f}")

# plot_liquidity_curve()
# slippage_analysis()
```

## Ключевые особенности AMM:

- **Доступность** - кто угодно может стать поставщиком ликвидности
- **Автоматизация** - цены определяются алгоритмически
- **Постоянная ликвидность** - торговля возможна в любое время
- **Имперманентная потеря** - риск для поставщиков ликвидности

## Популярные реализации:

- **Uniswap** (константный продукт)
- **Curve** (стабильные пулы)
- **Balancer** (пулы с несколькими токенами)
- **Bancor** (пулы с одной стороной)

