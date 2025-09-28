#!/usr/bin/env python3
"""
Исследование проблем со стоп-лоссами
Детальный анализ логики срабатывания стоп-лосса
"""
import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

from src.data.technical_indicators import vectorized_signal_generation

def analyze_stop_loss_logic():
    """Исследование логики стоп-лосса с реальными сценариями"""
    print("="*80)
    print("ИССЛЕДОВАНИЕ ЛОГИКИ СТОП-ЛОССА")
    print("="*80)

    # Создадим сценарий где стоп-лосс должен сработать
    prices = np.array([
        1.0000,  # Entry point (будет вход на нижний BB)
        0.9950,  # -0.5% (near stop loss)
        0.9940,  # -0.6% (должен сработать стоп)
        0.9930,  # -0.7% (продолжает падать)
        1.0100,  # +1.0% (восстановился)
    ])

    # SMA (средняя линия BB)
    sma = np.full(len(prices), 1.0000)

    # BB полосы
    upper_band = np.array([1.0200, 1.0200, 1.0200, 1.0200, 1.0200])  # +2%
    lower_band = np.array([0.9800, 0.9800, 0.9800, 0.9800, 0.9800])  # -2% (вход будет здесь)

    stop_loss_pct = 0.005  # 0.5% стоп-лосс

    print("\nТЕСТОВЫЕ ДАННЫЕ:")
    print(f"SMA (средняя): {sma[0]:.4f}")
    print(f"Upper Band: {upper_band[0]:.4f}")
    print(f"Lower Band: {lower_band[0]:.4f}")
    print(f"Stop Loss: {stop_loss_pct*100}%")

    print("\nЦЕНЫ:")
    for i, price in enumerate(prices):
        print(f"  {i}: {price:.4f}")

    # Модифицируем первый элемент чтобы создать вход на lower_band
    # Имитируем что цена пересекла нижнюю полосу
    prices[0] = 0.9850  # Выше нижней полосы
    prices[1] = 0.9790  # Ниже нижней полосы (вход)
    prices[2] = 0.9740  # -0.5% от entry (0.9790 * 0.995 = 0.9745, близко к стоп)
    prices[3] = 0.9730  # -0.6% от entry (должен сработать стоп)

    print("\nМОДИФИЦИРОВАННЫЕ ЦЕНЫ (для создания входа):")
    for i, price in enumerate(prices):
        print(f"  {i}: {price:.4f}")

    # Анализируем сигналы
    entry_signals, exit_signals, position_status = vectorized_signal_generation(
        prices, sma, upper_band, lower_band, stop_loss_pct
    )

    print("\n" + "="*80)
    print("АНАЛИЗ СТОП-ЛОССА:")
    print("="*80)

    for i in range(len(prices)):
        print(f"\nШаг {i}: Цена = {prices[i]:.4f}")

        if entry_signals[i] != 0:
            signal_type = "LONG" if entry_signals[i] == 1 else "SHORT"
            print(f"  *** ВХОД: {signal_type} (entry_signal = {entry_signals[i]})")

        if exit_signals[i] != 0:
            exit_type_map = {1: "СТОП-ЛОСС ЛОНГ", 2: "ТЕЙК-ПРОФИТ ЛОНГ", -1: "СТОП-ЛОСС ШОРТ", -2: "ТЕЙК-ПРОФИТ ШОРТ"}
            exit_type = exit_type_map.get(exit_signals[i], f"НЕИЗВЕСТНЫЙ ({exit_signals[i]})")
            print(f"  [X] ВЫХОД: {exit_type}")

            # Анализируем причину выхода
            if i > 0 and position_status[i-1] == 1:  # Был лонг
                prev_price = prices[i-1]
                current_price = prices[i]
                stop_loss_level = prev_price * (1 - stop_loss_pct)
                hit_sma = current_price >= sma[i]
                hit_stop = current_price <= stop_loss_level

                print(f"    Детальный анализ ЛОНГ выхода:")
                print(f"    - Цена входа (предполагаемая): {prices[i-2] if i >= 2 else 'N/A':.4f}")
                print(f"    - Предыдущая цена: {prev_price:.4f}")
                print(f"    - Текущая цена: {current_price:.4f}")
                print(f"    - Стоп-лосс от предыдущей цены: {stop_loss_level:.4f}")
                print(f"    - SMA уровень: {sma[i]:.4f}")
                print(f"    - Цена >= SMA? {hit_sma}")
                print(f"    - Цена <= Stop? {hit_stop}")

                if hit_stop:
                    print(f"    -> ПРИЧИНА: СТОП-ЛОСС (цена упала ниже {stop_loss_level:.4f})")
                elif hit_sma:
                    print(f"    -> ПРИЧИНА: ТЕЙК-ПРОФИТ (возврат к SMA {sma[i]:.4f})")

                print(f"    *** ПРОБЛЕМА: Стоп считается от предыдущей цены, а не от входа!")

        pos_desc = {0: "НЕТ ПОЗИЦИИ", 1: "ЛОНГ", -1: "ШОРТ"}
        print(f"  Позиция: {pos_desc[position_status[i]]}")

def analyze_real_stop_loss_issues():
    """Анализ реальных проблем со стоп-лоссом"""
    print("\n" + "="*80)
    print("АНАЛИЗ РЕАЛЬНЫХ ПРОБЛЕМ СО СТОП-ЛОССОМ:")
    print("="*80)

    print("\n1. ПРОБЛЕМА: Стоп-лосс от предыдущей цены, а не от входа")
    print("   ТЕКУЩАЯ ЛОГИКА:")
    print("   stop_loss = prices[i-1] * (1 - stop_loss_pct)")
    print("   ")
    print("   ПРАВИЛЬНАЯ ЛОГИКА должна быть:")
    print("   stop_loss = entry_price * (1 - stop_loss_pct)")

    print("\n2. ПРОБЛЕМА: Стоп-лосс может сработать сразу после входа")
    print("   Если цена немного колеблется после входа, может сразу сработать стоп")

    print("\n3. ПРОБЛЕМА: Нет сохранения цены входа")
    print("   В current_pos мы храним только 1/-1, но не цену входа")

    print("\n4. РЕШЕНИЕ:")
    print("   - Сохранять entry_price для каждой позиции")
    print("   - Считать стоп-лосс от entry_price, а не от предыдущей цены")
    print("   - Использовать отдельные массивы для отслеживания входов")

if __name__ == "__main__":
    analyze_stop_loss_logic()
    analyze_real_stop_loss_issues()