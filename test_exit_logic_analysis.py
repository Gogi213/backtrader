#!/usr/bin/env python3
"""
Тест для анализа логики тейк-профита и стоп-лосса
Детальный разбор условий выходов из позиций
"""
import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

from src.data.technical_indicators import vectorized_signal_generation

def analyze_exit_logic():
    """Детальный анализ логики выходов с конкретными примерами"""
    print("="*80)
    print("ДЕТАЛЬНЫЙ АНАЛИЗ ЛОГИКИ ТЕЙК-ПРОФИТА И СТОП-ЛОССА")
    print("="*80)

    # Создаем тестовые данные для анализа
    prices = np.array([
        1.0000,  # Entry point
        0.9950,  # -0.5% (near stop loss)
        0.9900,  # -1.0% (stop loss triggered)
        1.0050,  # +0.5%
        1.0100,  # +1.0% (target area)
        1.0000,  # Back to SMA (target hit)
        0.9800,  # Large drop
    ])

    # SMA (средняя линия BB)
    sma = np.full(len(prices), 1.0000)

    # BB полосы (2 std)
    upper_band = np.full(len(prices), 1.0200)  # +2%
    lower_band = np.full(len(prices), 0.9800)  # -2%

    stop_loss_pct = 0.005  # 0.5% стоп-лосс

    print("\nТЕСТОВЫЕ ДАННЫЕ:")
    print(f"SMA (средняя): {sma[0]:.4f}")
    print(f"Upper Band: {upper_band[0]:.4f} (+2%)")
    print(f"Lower Band: {lower_band[0]:.4f} (-2%)")
    print(f"Stop Loss: {stop_loss_pct*100}%")

    print("\nЦЕНЫ:")
    for i, price in enumerate(prices):
        print(f"  {i}: {price:.4f}")

    # Анализируем сигналы
    entry_signals, exit_signals, position_status = vectorized_signal_generation(
        prices, sma, upper_band, lower_band, stop_loss_pct
    )

    print("\n" + "="*80)
    print("АНАЛИЗ СИГНАЛОВ:")
    print("="*80)

    for i in range(len(prices)):
        print(f"\nШаг {i}: Цена = {prices[i]:.4f}")

        if entry_signals[i] != 0:
            signal_type = "LONG" if entry_signals[i] == 1 else "SHORT"
            print(f"  *** ВХОД: {signal_type} (entry_signal = {entry_signals[i]})")

        if exit_signals[i] != 0:
            exit_type = "LONG EXIT" if exit_signals[i] == 1 else "SHORT EXIT"
            print(f"  [X] ВЫХОД: {exit_type} (exit_signal = {exit_signals[i]})")

            # Анализируем причину выхода
            if i > 0:
                prev_price = prices[i-1]
                current_price = prices[i]

                if position_status[i-1] == 1:  # Был лонг
                    stop_loss_level = prev_price * (1 - stop_loss_pct)
                    hit_sma = current_price >= sma[i]
                    hit_stop = current_price <= stop_loss_level

                    print(f"    Анализ ЛОНГ выхода:")
                    print(f"    - Стоп-лосс уровень: {stop_loss_level:.4f}")
                    print(f"    - Цена >= SMA? {hit_sma} (SMA = {sma[i]:.4f})")
                    print(f"    - Цена <= Stop? {hit_stop}")

                    if hit_stop:
                        print(f"    -> ПРИЧИНА: СТОП-ЛОСС (-{stop_loss_pct*100}%)")
                    elif hit_sma:
                        print(f"    -> ПРИЧИНА: ТЕЙК-ПРОФИТ (возврат к средней)")

                elif position_status[i-1] == -1:  # Был шорт
                    stop_loss_level = prev_price * (1 + stop_loss_pct)
                    hit_sma = current_price <= sma[i]
                    hit_stop = current_price >= stop_loss_level

                    print(f"    Анализ ШОРТ выхода:")
                    print(f"    - Стоп-лосс уровень: {stop_loss_level:.4f}")
                    print(f"    - Цена <= SMA? {hit_sma} (SMA = {sma[i]:.4f})")
                    print(f"    - Цена >= Stop? {hit_stop}")

                    if hit_stop:
                        print(f"    -> ПРИЧИНА: СТОП-ЛОСС (+{stop_loss_pct*100}%)")
                    elif hit_sma:
                        print(f"    -> ПРИЧИНА: ТЕЙК-ПРОФИТ (возврат к средней)")

        pos_desc = {0: "НЕТ ПОЗИЦИИ", 1: "ЛОНГ", -1: "ШОРТ"}
        print(f"  Позиция: {pos_desc[position_status[i]]}")

def analyze_current_strategy_logic():
    """Анализ текущей логики в коде"""
    print("\n" + "="*80)
    print("АНАЛИЗ ЛОГИКИ В КОДЕ:")
    print("="*80)

    print("\n1. УСЛОВИЯ ВХОДА:")
    print("   ЛОНГ: prices[i+1] <= lower_band[i+1] AND prices[i] > lower_band[i]")
    print("         (цена пересекла нижнюю полосу сверху вниз)")
    print("   ШОРТ: prices[i+1] >= upper_band[i+1] AND prices[i] < upper_band[i]")
    print("         (цена пересекла верхнюю полосу снизу вверх)")

    print("\n2. УСЛОВИЯ ВЫХОДА ДЛЯ ЛОНГА:")
    print("   СТОП-ЛОСС: prices[i] <= prices[i-1] * (1 - stop_loss_pct)")
    print("   ТЕЙК-ПРОФИТ: prices[i] >= sma[i] (возврат к средней)")

    print("\n3. УСЛОВИЯ ВЫХОДА ДЛЯ ШОРТА:")
    print("   СТОП-ЛОСС: prices[i] >= prices[i-1] * (1 + stop_loss_pct)")
    print("   ТЕЙК-ПРОФИТ: prices[i] <= sma[i] (возврат к средней)")

    print("\n4. ОПРЕДЕЛЕНИЕ ПРИЧИНЫ ВЫХОДА:")
    print("   'stop_loss' if abs(exit_signal) == 1 else 'target_hit'")
    print("   - exit_signal == 1/-1: всегда (и стоп, и тейк)")
    print("   - ПРОБЛЕМА: Нельзя различить стоп от тейка!")

def create_improvement_plan():
    """План улучшений на основе анализа"""
    print("\n" + "="*80)
    print("ПЛАН УЛУЧШЕНИЙ:")
    print("="*80)

    print("\n1. ПРОБЛЕМЫ ТЕКУЩЕЙ СИСТЕМЫ:")
    print("   [ERROR] Нельзя различить стоп-лосс от тейк-профита")
    print("   [ERROR] exit_reason всегда 'target_hit' (abs(exit_signal) всегда == 1)")
    print("   [ERROR] Нет фиксированного тейк-профита, только возврат к средней")
    print("   [ERROR] Стоп-лосс вычисляется от предыдущей цены, а не от входа")

    print("\n2. ПРЕДЛАГАЕМЫЕ ИСПРАВЛЕНИЯ:")
    print("   [OK] Изменить exit_signals на разные значения:")
    print("      exit_signal = 1: стоп-лосс лонг")
    print("      exit_signal = 2: тейк-профит лонг (возврат к SMA)")
    print("      exit_signal = -1: стоп-лосс шорт")
    print("      exit_signal = -2: тейк-профит шорт (возврат к SMA)")

    print("\n   [OK] Добавить фиксированный тейк-профит:")
    print("      take_profit_pct = 1.0%  # Опционально")

    print("\n   [OK] Исправить расчет стоп-лосса от цены входа:")
    print("      entry_price * (1 ± stop_loss_pct)")

    print("\n3. ВИЗУАЛИЗАЦИЯ НА ГРАФИКЕ:")
    print("   [X] Стоп-лосс: Красный X")
    print("   [S] Тейк-профит (SMA): Зеленый квадрат")
    print("   [T] Тейк-профит (фикс): Желтый круг")
    print("   [-] Линии соединяющие вход и выход")

if __name__ == "__main__":
    analyze_exit_logic()
    analyze_current_strategy_logic()
    create_improvement_plan()