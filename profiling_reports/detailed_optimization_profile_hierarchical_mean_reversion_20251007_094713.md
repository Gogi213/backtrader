DETAILED OPTUNA PROFILING REPORT
============================================================

OPTIMIZATION ANALYSIS
-------------------------
Total trials: 15
Total time: 149.7493s
Average trial time: 9.9833s
Median trial time: 9.9110s
Min trial time: 9.4403s
Max trial time: 11.4929s
Std trial time: 0.4420s
Optimization efficiency: 97.93%

SLOWEST TRIALS
--------------------
1. Trial 0: 11.4929s
2. Trial 8: 10.1605s
3. Trial 7: 10.0766s
4. Trial 2: 10.0446s
5. Trial 1: 9.9985s

FASTEST TRIALS
--------------------
1. Trial 13: 9.8525s
2. Trial 10: 9.7926s
3. Trial 5: 9.7780s
4. Trial 12: 9.5472s
5. Trial 9: 9.4403s

FUNCTION HOTSPOTS
--------------------
SLOWEST FUNCTIONS
--------------------
DETAILED CPROFILE STATS
-------------------------
         4445171 function calls (4433070 primitive calls) in 9.911 seconds

   Ordered by: cumulative time
   List reduced from 638 to 50 due to restriction <50>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.003    0.003    9.911    9.911 C:\visual projects\backtrader\src\optimization\fast_optimizer.py:301(objective)
        1    0.009    0.009    9.718    9.718 C:\visual projects\backtrader\src\strategies\turbo_mean_reversion_strategy.py:507(turbo_process_dataset)
        1    0.000    0.000    8.871    8.871 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\pykalman\standard.py:1182(filter)
        1    0.272    0.272    8.870    8.870 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\pykalman\standard.py:308(_filter)
    12096    0.705    0.000    7.504    0.001 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\pykalman\standard.py:232(_filter_correct)
    24192    0.556    0.000    3.230    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\ma\core.py:1010(__call__)
24222/12126    0.317    0.000    2.716    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\scipy\_lib\_util.py:1203(wrapper)
    12096    0.559    0.000    2.510    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\scipy\linalg\_basic.py:1532(pinv)
    12096    0.048    0.000    1.760    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\ma\core.py:4228(__sub__)
    12096    0.027    0.000    1.583    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\ma\core.py:4219(__radd__)
    48386    0.492    0.000    1.365    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\ma\core.py:2978(__array_finalize__)
    84675    0.655    0.000    1.006    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\ma\core.py:2952(_update_from)
    72579    0.102    0.000    0.987    0.000 {method 'view' of 'numpy.ndarray' objects}
    12096    0.397    0.000    0.887    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\scipy\linalg\_decomp_svd.py:16(svd)
        1    0.045    0.045    0.733    0.733 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\sklearn\mixture\_base.py:393(predict_proba)
    12096    0.128    0.000    0.705    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\ma\core.py:3217(__getitem__)
        6    0.040    0.007    0.704    0.117 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\sklearn\mixture\_base.py:513(_estimate_log_prob_resp)
    72980    0.694    0.000    0.694    0.000 {method 'reduce' of 'numpy.ufunc' objects}
    36501    0.160    0.000    0.686    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\core\fromnumeric.py:71(_wrapreduction)
    72690    0.262    0.000    0.562    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\core\_ufunc_config.py:33(seterr)
        6    0.076    0.013    0.521    0.087 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\scipy\special\_logsumexp.py:15(logsumexp)
    24222    0.118    0.000    0.481    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\scipy\_lib\_util.py:410(_asarray_validated)
        6    0.188    0.031    0.419    0.070 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\scipy\special\_logsumexp.py:205(_logsumexp)
    24193    0.109    0.000    0.385    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\ma\core.py:1424(getmaskarray)
    12102    0.036    0.000    0.307    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\core\fromnumeric.py:2692(max)
    24249    0.053    0.000    0.307    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\core\_ufunc_config.py:430(__enter__)
    12213    0.043    0.000    0.302    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\core\fromnumeric.py:2177(sum)
   822861    0.242    0.000    0.242    0.000 {built-in method builtins.getattr}
    12141    0.088    0.000    0.229    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\lib\function_base.py:564(asarray_chkfinite)
    48384    0.145    0.000    0.223    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\ma\core.py:671(getdata)
    24193    0.045    0.000    0.219    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\ma\core.py:1644(make_mask_none)
    24192    0.040    0.000    0.215    0.000 {method 'any' of 'numpy.ndarray' objects}
    72690    0.195    0.000    0.212    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\core\_ufunc_config.py:132(geterr)
    12130    0.050    0.000    0.211    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\core\fromnumeric.py:2322(any)
    12096    0.129    0.000    0.207    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\scipy\linalg\lapack.py:1025(_compute_lwork)
    24249    0.053    0.000    0.204    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\core\_ufunc_config.py:435(__exit__)
    72573    0.142    0.000    0.200    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\pykalman\standard.py:92(_last_dims)
    24192    0.129    0.000    0.194    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\ma\core.py:644(get_masked_subclass)
   377566    0.144    0.000    0.191    0.000 {built-in method builtins.isinstance}
    12095    0.177    0.000    0.189    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\pykalman\standard.py:186(_filter_predict)
       16    0.000    0.000    0.186    0.012 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\optuna\trial\_trial.py:611(_suggest)
       16    0.000    0.000    0.183    0.011 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\optuna\samplers\_tpe\sampler.py:395(sample_independent)
       16    0.001    0.000    0.183    0.011 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\optuna\samplers\_tpe\sampler.py:437(_sample)
    24193    0.027    0.000    0.175    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\core\_methods.py:55(_any)
       14    0.000    0.000    0.157    0.011 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\optuna\trial\_trial.py:76(suggest_float)
        6    0.021    0.004    0.143    0.024 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\sklearn\mixture\_base.py:474(_estimate_weighted_log_prob)
   375025    0.135    0.000    0.135    0.000 {method 'update' of 'dict' objects}
   133059    0.094    0.000    0.133    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\ma\core.py:1362(getmask)
    12153    0.022    0.000    0.133    0.000 {method 'all' of 'numpy.ndarray' objects}
    24194    0.026    0.000    0.132    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\ma\core.py:1329(make_mask_descr)


