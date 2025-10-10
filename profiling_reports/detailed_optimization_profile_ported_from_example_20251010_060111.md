OPTUNA OPTIMIZATION PROFILING REPORT
==================================================

## Study Summary

- Total Trials: 30
- Successful: 30
- Pruned: 0
- Failed: 0
- Total Profiling Time: 366.10s

## Timing Analysis (for completed trials)

- Average Trial Time: 12.2033s
- Median Trial Time: 9.4851s
- Min/Max Trial Time: 0.9880s / 27.6858s

## Slowest Trials

Use `profiler.get_trial_report(trial_number)` for a detailed breakdown.

  1. Trial 1: 27.6858s
  2. Trial 7: 27.5758s
  3. Trial 3: 25.8418s
  4. Trial 0: 25.0548s
  5. Trial 6: 24.9557s

## Full Data (Summary)

```json
{
  "5": {
    "execution_time": 24.364766359329224,
    "result": -Infinity,
    "params": {
      "vol_period": 95,
      "vol_pctl": 6.95,
      "range_period": 39,
      "rng_pctl": 1.07,
      "min_growth_pct": 2.29,
      "entry_logic_mode": "\u0422\u043e\u043b\u044c\u043a\u043e \u043f\u043e \u043f\u0440\u0438\u043d\u0442\u0430\u043c",
      "prints_analysis_period": 3,
      "prints_threshold_ratio": 3.89,
      "hldir_window": 10,
      "hldir_offset": 5,
      "stop_loss_pct": 5.53,
      "take_profit_pct": 14.9,
      "aggressive_mode": true
    },
    "state": "COMPLETE"
  },
  "2": {
    "execution_time": 24.406760215759277,
    "result": -Infinity,
    "params": {
      "vol_period": 60,
      "vol_pctl": 8.129999999999999,
      "range_period": 66,
      "rng_pctl": 7.66,
      "min_growth_pct": 3.5900000000000003,
      "entry_logic_mode": "\u0422\u043e\u043b\u044c\u043a\u043e \u043f\u043e HLdir",
      "prints_analysis_period": 10,
      "prints_threshold_ratio": 4.050000000000001,
      "hldir_window": 13,
      "hldir_offset": 2,
      "stop_loss_pct": 3.8000000000000003,
      "take_profit_pct": 17.02,
      "aggressive_mode": false
    },
    "state": "COMPLETE"
  },
  "6": {
    "execution_time": 24.955745697021484,
    "result": -Infinity,
    "params": {
      "vol_period": 42,
      "vol_pctl": 1.9800000000000002,
      "range_period": 14,
      "rng_pctl": 1.12,
      "min_growth_pct": 3.04,
      "entry_logic_mode": "\u0422\u043e\u043b\u044c\u043a\u043e \u043f\u043e HLdir",
      "prints_analysis_period": 10,
      "prints_threshold_ratio": 3.08,
      "hldir_window": 10,
      "hldir_offset": 0,
      "stop_loss_pct": 8.21,
      "take_profit_pct": 1.03,
      "aggressive_mode": false
    },
    "state": "COMPLETE"
  },
  "4": {
    "execution_time": 24.399728298187256,
    "result": -Infinity,
    "params": {
      "vol_period": 54,
      "vol_pctl": 7.41,
      "range_period": 31,
      "rng_pctl": 9.19,
      "min_growth_pct": 0.43000000000000005,
      "entry_logic_mode": "\u0422\u043e\u043b\u044c\u043a\u043e \u043f\u043e HLdir",
      "prints_analysis_period": 8,
      "prints_threshold_ratio": 2.9800000000000004,
      "hldir_window": 7,
      "hldir_offset": 4,
      "stop_loss_pct": 3.81,
      "take_profit_pct": 9.38,
      "aggressive_mode": false
    },
    "state": "COMPLETE"
  },
  "3": {
    "execution_time": 25.841784477233887,
    "result": -Infinity,
    "params": {
      "vol_period": 64,
      "vol_pctl": 1.55,
      "range_period": 11,
      "rng_pctl": 7.41,
      "min_growth_pct": 2.3200000000000003,
      "entry_logic_mode": "\u0422\u043e\u043b\u044c\u043a\u043e \u043f\u043e \u043f\u0440\u0438\u043d\u0442\u0430\u043c",
      "prints_analysis_period": 5,
      "prints_threshold_ratio": 2.75,
      "hldir_window": 17,
      "hldir_offset": 1,
      "stop_loss_pct": 6.92,
      "take_profit_pct": 2.24,
      "aggressive_mode": true
    },
    "state": "COMPLETE"
  },
  "0": {
    "execution_time": 25.054779767990112,
    "result": -Infinity,
    "params": {
      "vol_period": 51,
      "vol_pctl": 5.63,
      "range_period": 57,
      "rng_pctl": 3.6300000000000003,
      "min_growth_pct": 3.48,
      "entry_logic_mode": "\u041f\u0440\u0438\u043d\u0442\u044b \u0438 HLdir",
      "prints_analysis_period": 9,
      "prints_threshold_ratio": 2.75,
      "hldir_window": 8,
      "hldir_offset": 5,
      "stop_loss_pct": 9.01,
      "take_profit_pct": 13.0,
      "aggressive_mode": true
    },
    "state": "COMPLETE"
  },
  "1": {
    "execution_time": 27.685774087905884,
    "result": -Infinity,
    "params": {
      "vol_period": 29,
      "vol_pctl": 3.5300000000000002,
      "range_period": 12,
      "rng_pctl": 8.72,
      "min_growth_pct": 0.21000000000000002,
      "entry_logic_mode": "\u0422\u043e\u043b\u044c\u043a\u043e \u043f\u043e \u043f\u0440\u0438\u043d\u0442\u0430\u043c",
      "prints_analysis_period": 5,
      "prints_threshold_ratio": 2.46,
      "hldir_window": 9,
      "hldir_offset": 3,
      "stop_loss_pct": 8.04,
      "take_profit_pct": 6.7,
      "aggressive_mode": true
    },
    "state": "COMPLETE"
  },
  "7": {
    "execution_time": 27.57577419281006,
    "result": -Infinity,
    "params": {
      "vol_period": 52,
      "vol_pctl": 3.68,
      "range_period": 85,
      "rng_pctl": 4.199999999999999,
      "min_growth_pct": 5.0,
      "entry_logic_mode": "\u0422\u043e\u043b\u044c\u043a\u043e \u043f\u043e HLdir",
      "prints_analysis_period": 1,
      "prints_threshold_ratio": 4.4399999999999995,
      "hldir_window": 14,
      "hldir_offset": 2,
      "stop_loss_pct": 8.46,
      "take_profit_pct": 1.07,
      "aggressive_mode": false
    },
    "state": "COMPLETE"
  },
  "8": {
    "execution_time": 1.605018138885498,
    "result": -Infinity,
    "params": {
      "vol_period": 33,
      "vol_pctl": 1.6,
      "range_period": 21,
      "rng_pctl": 3.3200000000000003,
      "min_growth_pct": 4.88,
      "entry_logic_mode": "\u0422\u043e\u043b\u044c\u043a\u043e \u043f\u043e \u043f\u0440\u0438\u043d\u0442\u0430\u043c",
      "prints_analysis_period": 9,
      "prints_threshold_ratio": 1.1600000000000001,
      "hldir_window": 8,
      "hldir_offset": 4,
      "stop_loss_pct": 6.3,
      "take_profit_pct": 9.76,
      "aggressive_mode": false
    },
    "state": "COMPLETE"
  },
  "9": {
    "execution_time": 0.988004207611084,
    "result": -Infinity,
    "params": {
      "vol_period": 25,
      "vol_pctl": 8.49,
      "range_period": 73,
      "rng_pctl": 2.23,
      "min_growth_pct": 2.16,
      "entry_logic_mode": "\u041f\u0440\u0438\u043d\u0442\u044b \u0438 HLdir",
      "prints_analysis_period": 6,
      "prints_threshold_ratio": 2.16,
      "hldir_window": 7,
      "hldir_offset": 0,
      "stop_loss_pct": 7.3500000000000005,
      "take_profit_pct": 4.359999999999999,
      "aggressive_mode": false
    },
    "state": "COMPLETE"
  },
  "10": {
    "execution_time": 1.0780084133148193,
    "result": -Infinity,
    "params": {
      "vol_period": 77,
      "vol_pctl": 7.3999999999999995,
      "range_period": 22,
      "rng_pctl": 1.83,
      "min_growth_pct": 1.4600000000000002,
      "entry_logic_mode": "\u0422\u043e\u043b\u044c\u043a\u043e \u043f\u043e HLdir",
      "prints_analysis_period": 9,
      "prints_threshold_ratio": 3.3800000000000003,
      "hldir_window": 6,
      "hldir_offset": 2,
      "stop_loss_pct": 6.65,
      "take_profit_pct": 6.29,
      "aggressive_mode": false
    },
    "state": "COMPLETE"
  },
  "11": {
    "execution_time": 2.1433587074279785,
    "result": -Infinity,
    "params": {
      "vol_period": 19,
      "vol_pctl": 2.79,
      "range_period": 14,
      "rng_pctl": 7.3,
      "min_growth_pct": 2.5500000000000003,
      "entry_logic_mode": "\u0422\u043e\u043b\u044c\u043a\u043e \u043f\u043e HLdir",
      "prints_analysis_period": 8,
      "prints_threshold_ratio": 3.31,
      "hldir_window": 10,
      "hldir_offset": 2,
      "stop_loss_pct": 7.75,
      "take_profit_pct": 15.07,
      "aggressive_mode": true
    },
    "state": "COMPLETE"
  },
  "13": {
    "execution_time": 4.836902141571045,
    "result": -Infinity,
    "params": {
      "vol_period": 97,
      "vol_pctl": 4.51,
      "range_period": 66,
      "rng_pctl": 9.61,
      "min_growth_pct": 1.9400000000000002,
      "entry_logic_mode": "\u0422\u043e\u043b\u044c\u043a\u043e \u043f\u043e \u043f\u0440\u0438\u043d\u0442\u0430\u043c",
      "prints_analysis_period": 4,
      "prints_threshold_ratio": 2.3600000000000003,
      "hldir_window": 8,
      "hldir_offset": 4,
      "stop_loss_pct": 7.7700000000000005,
      "take_profit_pct": 12.57,
      "aggressive_mode": false
    },
    "state": "COMPLETE"
  },
  "12": {
    "execution_time": 4.8479108810424805,
    "result": -Infinity,
    "params": {
      "vol_period": 70,
      "vol_pctl": 9.91,
      "range_period": 53,
      "rng_pctl": 6.319999999999999,
      "min_growth_pct": 3.5,
      "entry_logic_mode": "\u0422\u043e\u043b\u044c\u043a\u043e \u043f\u043e \u043f\u0440\u0438\u043d\u0442\u0430\u043c",
      "prints_analysis_period": 7,
      "prints_threshold_ratio": 2.4800000000000004,
      "hldir_window": 18,
      "hldir_offset": 2,
      "stop_loss_pct": 4.32,
      "take_profit_pct": 17.78,
      "aggressive_mode": true
    },
    "state": "COMPLETE"
  },
  "14": {
    "execution_time": 7.038936138153076,
    "result": -Infinity,
    "params": {
      "vol_period": 10,
      "vol_pctl": 5.72,
      "range_period": 100,
      "rng_pctl": 5.77,
      "min_growth_pct": 3.79,
      "entry_logic_mode": "\u041f\u0440\u0438\u043d\u0442\u044b \u0438 HLdir",
      "prints_analysis_period": 7,
      "prints_threshold_ratio": 1.56,
      "hldir_window": 2,
      "hldir_offset": 5,
      "stop_loss_pct": 9.790000000000001,
      "take_profit_pct": 19.96,
      "aggressive_mode": true
    },
    "state": "COMPLETE"
  },
  "17": {
    "execution_time": 7.252202749252319,
    "result": -Infinity,
    "params": {
      "vol_period": 11,
      "vol_pctl": 5.67,
      "range_period": 99,
      "rng_pctl": 5.51,
      "min_growth_pct": 3.8400000000000003,
      "entry_logic_mode": "\u041f\u0440\u0438\u043d\u0442\u044b \u0438 HLdir",
      "prints_analysis_period": 7,
      "prints_threshold_ratio": 1.5,
      "hldir_window": 2,
      "hldir_offset": 5,
      "stop_loss_pct": 1.01,
      "take_profit_pct": 13.370000000000001,
      "aggressive_mode": true
    },
    "state": "COMPLETE"
  },
  "16": {
    "execution_time": 8.630942583084106,
    "result": -Infinity,
    "params": {
      "vol_period": 80,
      "vol_pctl": 5.59,
      "range_period": 50,
      "rng_pctl": 5.39,
      "min_growth_pct": 3.93,
      "entry_logic_mode": "\u041f\u0440\u0438\u043d\u0442\u044b \u0438 HLdir",
      "prints_analysis_period": 7,
      "prints_threshold_ratio": 4.98,
      "hldir_window": 2,
      "hldir_offset": 5,
      "stop_loss_pct": 0.8400000000000001,
      "take_profit_pct": 13.11,
      "aggressive_mode": true
    },
    "state": "COMPLETE"
  },
  "15": {
    "execution_time": 9.932468891143799,
    "result": -Infinity,
    "params": {
      "vol_period": 80,
      "vol_pctl": 5.739999999999999,
      "range_period": 50,
      "rng_pctl": 5.96,
      "min_growth_pct": 3.85,
      "entry_logic_mode": "\u041f\u0440\u0438\u043d\u0442\u044b \u0438 HLdir",
      "prints_analysis_period": 7,
      "prints_threshold_ratio": 1.69,
      "hldir_window": 2,
      "hldir_offset": 5,
      "stop_loss_pct": 0.56,
      "take_profit_pct": 13.56,
      "aggressive_mode": true
    },
    "state": "COMPLETE"
  },
  "18": {
    "execution_time": 11.118918657302856,
    "result": -Infinity,
    "params": {
      "vol_period": 14,
      "vol_pctl": 4.859999999999999,
      "range_period": 50,
      "rng_pctl": 5.8,
      "min_growth_pct": 0.12000000000000001,
      "entry_logic_mode": "\u041f\u0440\u0438\u043d\u0442\u044b \u0438 HLdir",
      "prints_analysis_period": 6,
      "prints_threshold_ratio": 2.0700000000000003,
      "hldir_window": 2,
      "hldir_offset": 5,
      "stop_loss_pct": 9.88,
      "take_profit_pct": 12.52,
      "aggressive_mode": true
    },
    "state": "COMPLETE"
  },
  "21": {
    "execution_time": 8.880603790283203,
    "result": -Infinity,
    "params": {
      "vol_period": 41,
      "vol_pctl": 5.63,
      "range_period": 50,
      "rng_pctl": 5.34,
      "min_growth_pct": 0.25,
      "entry_logic_mode": "\u041f\u0440\u0438\u043d\u0442\u044b \u0438 HLdir",
      "prints_analysis_period": 3,
      "prints_threshold_ratio": 2.1100000000000003,
      "hldir_window": 2,
      "hldir_offset": 5,
      "stop_loss_pct": 9.8,
      "take_profit_pct": 7.17,
      "aggressive_mode": true
    },
    "state": "COMPLETE"
  },
  "19": {
    "execution_time": 9.88761830329895,
    "result": -Infinity,
    "params": {
      "vol_period": 39,
      "vol_pctl": 0.11,
      "range_period": 48,
      "rng_pctl": 4.91,
      "min_growth_pct": 0.13,
      "entry_logic_mode": "\u041f\u0440\u0438\u043d\u0442\u044b \u0438 HLdir",
      "prints_analysis_period": 3,
      "prints_threshold_ratio": 1.6700000000000002,
      "hldir_window": 2,
      "hldir_offset": 5,
      "stop_loss_pct": 9.870000000000001,
      "take_profit_pct": 7.47,
      "aggressive_mode": true
    },
    "state": "COMPLETE"
  },
  "20": {
    "execution_time": 10.78211498260498,
    "result": -Infinity,
    "params": {
      "vol_period": 40,
      "vol_pctl": 0.11,
      "range_period": 47,
      "rng_pctl": 5.38,
      "min_growth_pct": 0.1,
      "entry_logic_mode": "\u041f\u0440\u0438\u043d\u0442\u044b \u0438 HLdir",
      "prints_analysis_period": 3,
      "prints_threshold_ratio": 1.82,
      "hldir_window": 4,
      "hldir_offset": 5,
      "stop_loss_pct": 9.71,
      "take_profit_pct": 6.97,
      "aggressive_mode": true
    },
    "state": "COMPLETE"
  },
  "22": {
    "execution_time": 10.524319171905518,
    "result": -Infinity,
    "params": {
      "vol_period": 41,
      "vol_pctl": 5.45,
      "range_period": 47,
      "rng_pctl": 4.37,
      "min_growth_pct": 0.94,
      "entry_logic_mode": "\u041f\u0440\u0438\u043d\u0442\u044b \u0438 HLdir",
      "prints_analysis_period": 3,
      "prints_threshold_ratio": 1.9900000000000002,
      "hldir_window": 3,
      "hldir_offset": 3,
      "stop_loss_pct": 1.28,
      "take_profit_pct": 7.42,
      "aggressive_mode": true
    },
    "state": "COMPLETE"
  },
  "23": {
    "execution_time": 7.866826057434082,
    "result": -Infinity,
    "params": {
      "vol_period": 42,
      "vol_pctl": 0.6,
      "range_period": 47,
      "rng_pctl": 4.05,
      "min_growth_pct": 0.25,
      "entry_logic_mode": "\u041f\u0440\u0438\u043d\u0442\u044b \u0438 HLdir",
      "prints_analysis_period": 2,
      "prints_threshold_ratio": 1.9500000000000002,
      "hldir_window": 4,
      "hldir_offset": 3,
      "stop_loss_pct": 9.84,
      "take_profit_pct": 7.22,
      "aggressive_mode": true
    },
    "state": "COMPLETE"
  },
  "25": {
    "execution_time": 8.544827699661255,
    "result": -Infinity,
    "params": {
      "vol_period": 41,
      "vol_pctl": 3.95,
      "range_period": 38,
      "rng_pctl": 4.01,
      "min_growth_pct": 0.53,
      "entry_logic_mode": "\u041f\u0440\u0438\u043d\u0442\u044b \u0438 HLdir",
      "prints_analysis_period": 3,
      "prints_threshold_ratio": 2.2,
      "hldir_window": 13,
      "hldir_offset": 3,
      "stop_loss_pct": 9.88,
      "take_profit_pct": 7.3500000000000005,
      "aggressive_mode": true
    },
    "state": "COMPLETE"
  },
  "24": {
    "execution_time": 9.512808799743652,
    "result": -Infinity,
    "params": {
      "vol_period": 42,
      "vol_pctl": 3.4200000000000004,
      "range_period": 39,
      "rng_pctl": 3.73,
      "min_growth_pct": 0.51,
      "entry_logic_mode": "\u041f\u0440\u0438\u043d\u0442\u044b \u0438 HLdir",
      "prints_analysis_period": 3,
      "prints_threshold_ratio": 2.0300000000000002,
      "hldir_window": 5,
      "hldir_offset": 3,
      "stop_loss_pct": 10.0,
      "take_profit_pct": 7.28,
      "aggressive_mode": true
    },
    "state": "COMPLETE"
  },
  "26": {
    "execution_time": 8.865801811218262,
    "result": -Infinity,
    "params": {
      "vol_period": 38,
      "vol_pctl": 0.12000000000000001,
      "range_period": 36,
      "rng_pctl": 4.13,
      "min_growth_pct": 0.92,
      "entry_logic_mode": "\u0422\u043e\u043b\u044c\u043a\u043e \u043f\u043e \u043f\u0440\u0438\u043d\u0442\u0430\u043c",
      "prints_analysis_period": 3,
      "prints_threshold_ratio": 3.58,
      "hldir_window": 5,
      "hldir_offset": 3,
      "stop_loss_pct": 9.02,
      "take_profit_pct": 7.53,
      "aggressive_mode": true
    },
    "state": "COMPLETE"
  },
  "28": {
    "execution_time": 8.505393505096436,
    "result": -Infinity,
    "params": {
      "vol_period": 60,
      "vol_pctl": 8.98,
      "range_period": 66,
      "rng_pctl": 8.62,
      "min_growth_pct": 1.1,
      "entry_logic_mode": "\u0422\u043e\u043b\u044c\u043a\u043e \u043f\u043e HLdir",
      "prints_analysis_period": 10,
      "prints_threshold_ratio": 3.8200000000000003,
      "hldir_window": 14,
      "hldir_offset": 3,
      "stop_loss_pct": 2.9,
      "take_profit_pct": 17.35,
      "aggressive_mode": false
    },
    "state": "COMPLETE"
  },
  "27": {
    "execution_time": 9.46830940246582,
    "result": -Infinity,
    "params": {
      "vol_period": 60,
      "vol_pctl": 9.02,
      "range_period": 64,
      "rng_pctl": 8.26,
      "min_growth_pct": 1.03,
      "entry_logic_mode": "\u0422\u043e\u043b\u044c\u043a\u043e \u043f\u043e HLdir",
      "prints_analysis_period": 10,
      "prints_threshold_ratio": 3.8200000000000003,
      "hldir_window": 14,
      "hldir_offset": 3,
      "stop_loss_pct": 2.7,
      "take_profit_pct": 16.79,
      "aggressive_mode": false
    },
    "state": "COMPLETE"
  },
  "29": {
    "execution_time": 9.50195050239563,
    "result": -Infinity,
    "params": {
      "vol_period": 59,
      "vol_pctl": 8.94,
      "range_period": 62,
      "rng_pctl": 8.51,
      "min_growth_pct": 0.91,
      "entry_logic_mode": "\u0422\u043e\u043b\u044c\u043a\u043e \u043f\u043e \u043f\u0440\u0438\u043d\u0442\u0430\u043c",
      "prints_analysis_period": 10,
      "prints_threshold_ratio": 4.0,
      "hldir_window": 14,
      "hldir_offset": 3,
      "stop_loss_pct": 2.64,
      "take_profit_pct": 17.79,
      "aggressive_mode": false
    },
    "state": "COMPLETE"
  }
}
```
