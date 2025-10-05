---
tags: [Экономика и финансы]
---

# Оракулы в блокчейне и UMA Protocol

**Оракулы** — это мосты между блокчейном и внешним миром. Они предоставляют смарт-контрактам доступ к реальным данным (цены активов, погода, результаты спортивных событий и т.д.).

**Проблема, которую решают оракулы:**

- Блокчейны изолированы и не могут напрямую получать внешние данные
- Смарт-контракты нуждаются в надежных внешних данных для выполнения условий

## Типы оракулов:

- **Централизованные** (например, Chainlink)
- **Децентрализованные** (например, UMA)
- **Software oracles** (данные из онлайн-источников)
- **Hardware oracles** (данные с физических устройств)

## UMA Protocol (Universal Market Access)

### Основные концепции:

- **Оптимистическая модель** - данные считаются верными, пока не оспорены
- **Экономические гарантии** - участники ставят залог для обеспечения честности
- **Система разрешения споров** - механизм для оспаривания некорректных данных

### Ключевые компоненты:

- **Data Verification Mechanism (DVM)** - децентрализованный оракул
- **Price Feeds** - ценовые фиды
- **LSP (Liquidator and Settler)** - ликвидаторы и расчетные модули

## Примеры использования UMA

### 1. Синтетические активы

```python
# Пример взаимодействия с UMA для создания синтетического актива
from web3 import Web3
import json

class UMASyntheticAsset:
    def __init__(self, provider_url, contract_address, abi_path):
        self.w3 = Web3(Web3.HTTPProvider(provider_url))
        with open(abi_path, 'r') as f:
            abi = json.load(f)
        self.contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(contract_address),
            abi=abi
        )

    def create_position(self, collateral_amount, synthetic_tokens):
        """Создание позиции синтетического актива"""
        try:
            tx = self.contract.functions.create(
                collateral_amount,
                synthetic_tokens
            ).build_transaction({
                'from': self.w3.eth.default_account,
                'gas': 2000000,
                'gasPrice': self.w3.eth.gas_price
            })
            return tx
        except Exception as e:
            print(f"Error creating position: {e}")
            return None

    def get_position_details(self, position_id):
        """Получение деталей позиции"""
        return self.contract.functions.positions(position_id).call()
```

### 2. Price Feed Consumer

```python
import requests
from typing import Dict, Any

class UMAPriceFeed:
    def __init__(self, uma_rpc_url: str):
        self.rpc_url = uma_rpc_url
        self.price_feeds = {
            'ETH/USD': '0x...',  # Адрес контракта price feed
            'BTC/USD': '0x...',
            'UMA/USD': '0x...'
        }

    def get_current_price(self, pair: str) -> float:
        """Получение текущей цены через UMA DVM"""
        if pair not in self.price_feeds:
            raise ValueError(f"Price feed for {pair} not found")

        # Здесь будет вызов контракта UMA Price Feed
        # Для примера используем упрощенный подход
        payload = {
            "jsonrpc": "2.0",
            "method": "eth_call",
            "params": [{
                "to": self.price_feeds[pair],
                "data": "0x50d25bcd"  # latestAnswer() метод
            }, "latest"],
            "id": 1
        }

        response = requests.post(self.rpc_url, json=payload)
        result = response.json()

        if 'result' in result:
            # Конвертируем hex в decimal
            price_hex = result['result']
            price = int(price_hex, 16) / 10**8  # UMA использует 8 decimal places
            return price

        raise Exception("Failed to fetch price from UMA")

    def get_historical_price(self, pair: str, timestamp: int) -> float:
        """Получение исторической цены"""
        # UMA DVM хранит исторические данные для разрешения споров
        payload = {
            "jsonrpc": "2.0",
            "method": "eth_call",
            "params": [{
                "to": self.price_feeds[pair],
                "data": f"0x...{timestamp:064x}"  # getHistoricalPrice метод
            }, "latest"],
            "id": 1
        }

        response = requests.post(self.rpc_url, json=payload)
        # Обработка ответа аналогично get_current_price
```

### 3. Dispute Resolution System

```python
class UMADisputeResolver:
    def __init__(self, web3_provider, uma_contract_address):
        self.w3 = web3_provider
        self.uma_contract = self.load_uma_contract(uma_contract_address)

    def dispute_price(self, price_identifier, timestamp, proposed_price):
        """Инициация спора о цене"""
        try:
            # Для оспаривания цены нужно внести залог
            dispute_fee = self.uma_contract.functions.getDisputeFee().call()

            tx = self.uma_contract.functions.disputePrice(
                price_identifier,
                timestamp,
                proposed_price
            ).build_transaction({
                'from': self.w3.eth.default_account,
                'value': dispute_fee,
                'gas': 500000,
                'gasPrice': self.w3.eth.gas_price
            })

            return tx
        except Exception as e:
            print(f"Error disputing price: {e}")
            return None

    def vote_on_dispute(self, dispute_id, support):
        """Голосование по спору"""
        return self.uma_contract.functions.vote(
            dispute_id,
            support
        ).build_transaction({
            'from': self.w3.eth.default_account,
            'gas': 300000,
            'gasPrice': self.w3.eth.gas_price
        })
```

### 4. Пример полного цикла работы с UMA

```python
class UMAIntegrationExample:
    def __init__(self):
        self.w3 = Web3(Web3.HTTPProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'))
        self.price_feed = UMAPriceFeed('https://mainnet.infura.io/v3/YOUR_PROJECT_ID')
        self.dispute_resolver = UMADisputeResolver(
            self.w3,
            '0x...'  # UMA contract address
        )

    def monitor_and_verify_prices(self):
        """Мониторинг и верификация цен через UMA"""

        # Получаем текущие цены
        eth_price = self.price_feed.get_current_price('ETH/USD')
        print(f"Current ETH/USD price: ${eth_price}")

        # Проверяем расхождения с другими источниками
        external_price = self.get_external_price('ETH/USD')

        # Если есть значительное расхождение, можем оспорить цену
        if abs(eth_price - external_price) / external_price > 0.05:  # 5% расхождение
            print("Significant price discrepancy detected!")

            # Оспариваем цену в UMA DVM
            dispute_tx = self.dispute_resolver.dispute_price(
                'ETH/USD',
                int(self.w3.eth.get_block('latest')['timestamp']),
                int(external_price * 10**8)  # Конвертируем в формат UMA
            )

            if dispute_tx:
                print("Dispute transaction prepared")
                # signed_tx = self.w3.eth.account.sign_transaction(dispute_tx, private_key)
                # tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)

    def get_external_price(self, pair: str) -> float:
        """Получение цены из внешнего источника для сравнения"""
        # Например, из CoinGecko API
        url = f"https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd"
        response = requests.get(url)
        data = response.json()
        return data['ethereum']['usd']
```

## Преимущества UMA:

- **Децентрализация** - нет единой точки отказа
- **Экономическая безопасность** - участники мотивированы действовать честно
- **Гибкость** - можно создавать различные финансовые продукты
- **Прозрачность** - все операции видны в блокчейне

## Недостатки:

- **Сложность** - требует понимания механизмов обеспечения безопасности
- **Время разрешения споров** - процесс оспаривания может занимать время
- **Требует залогов** - участники должны блокировать средства

## Практическое применение:

UMA особенно полезен для:

- Деривативов и синтетических активов
- Страхования
- Предсказательных рынков
- Любых приложений, требующих надежных внешних данных
