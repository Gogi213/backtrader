DETAILED OPTUNA PROFILING REPORT
============================================================

OPTIMIZATION ANALYSIS
-------------------------
Total trials: 15
Total time: 136.9336s
Average trial time: 9.1289s
Median trial time: 8.5163s
Min trial time: 8.2359s
Max trial time: 12.8649s
Std trial time: 1.3774s
Optimization efficiency: 97.67%

SLOWEST TRIALS
--------------------
1. Trial 0: 12.8649s
2. Trial 4: 12.2274s
3. Trial 5: 9.2771s
4. Trial 1: 9.1021s
5. Trial 12: 8.8733s

FASTEST TRIALS
--------------------
1. Trial 14: 8.4230s
2. Trial 8: 8.3942s
3. Trial 2: 8.3656s
4. Trial 9: 8.2589s
5. Trial 7: 8.2359s

FUNCTION HOTSPOTS
--------------------
SLOWEST FUNCTIONS
--------------------
DETAILED CPROFILE STATS
-------------------------
         4435990 function calls (4423889 primitive calls) in 8.423 seconds

   Ordered by: cumulative time
   List reduced from 622 to 50 due to restriction <50>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    8.423    8.423 C:\visual projects\backtrader\src\optimization\fast_optimizer.py:301(objective)
        1    0.003    0.003    8.211    8.211 C:\visual projects\backtrader\src\strategies\turbo_mean_reversion_strategy.py:502(turbo_process_dataset)
        1    0.000    0.000    8.122    8.122 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\pykalman\standard.py:1182(filter)
        1    0.246    0.246    8.121    8.121 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\pykalman\standard.py:308(_filter)
    12096    0.641    0.000    6.870    0.001 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\pykalman\standard.py:232(_filter_correct)
    24192    0.511    0.000    3.022    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\ma\core.py:1010(__call__)
24222/12126    0.288    0.000    2.429    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\scipy\_lib\_util.py:1203(wrapper)
    12096    0.497    0.000    2.243    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\scipy\linalg\_basic.py:1532(pinv)
    12096    0.042    0.000    1.647    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\ma\core.py:4228(__sub__)
    12096    0.026    0.000    1.480    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\ma\core.py:4219(__radd__)
    48386    0.421    0.000    1.251    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\ma\core.py:2978(__array_finalize__)
    84675    0.616    0.000    0.956    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\ma\core.py:2952(_update_from)
    72579    0.096    0.000    0.904    0.000 {method 'view' of 'numpy.ndarray' objects}
    12096    0.349    0.000    0.780    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\scipy\linalg\_decomp_svd.py:16(svd)
    12096    0.107    0.000    0.629    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\ma\core.py:3217(__getitem__)
    72690    0.254    0.000    0.544    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\core\_ufunc_config.py:33(seterr)
    36500    0.151    0.000    0.494    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\core\fromnumeric.py:71(_wrapreduction)
    72977    0.484    0.000    0.484    0.000 {method 'reduce' of 'numpy.ufunc' objects}
    24222    0.111    0.000    0.429    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\scipy\_lib\_util.py:410(_asarray_validated)
    24193    0.101    0.000    0.352    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\ma\core.py:1424(getmaskarray)
    24249    0.047    0.000    0.293    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\core\_ufunc_config.py:430(__enter__)
   822857    0.235    0.000    0.235    0.000 {built-in method builtins.getattr}
    12102    0.035    0.000    0.225    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\core\fromnumeric.py:2692(max)
    72690    0.191    0.000    0.208    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\core\_ufunc_config.py:132(geterr)
       16    0.000    0.000    0.208    0.013 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\optuna\trial\_trial.py:611(_suggest)
       16    0.000    0.000    0.205    0.013 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\optuna\samplers\_tpe\sampler.py:395(sample_independent)
       16    0.001    0.000    0.204    0.013 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\optuna\samplers\_tpe\sampler.py:437(_sample)
    48384    0.132    0.000    0.203    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\ma\core.py:671(getdata)
    24249    0.051    0.000    0.198    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\core\_ufunc_config.py:435(__exit__)
    12213    0.040    0.000    0.198    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\core\fromnumeric.py:2177(sum)
    24192    0.036    0.000    0.197    0.000 {method 'any' of 'numpy.ndarray' objects}
    24193    0.040    0.000    0.197    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\ma\core.py:1644(make_mask_none)
    72573    0.140    0.000    0.197    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\pykalman\standard.py:92(_last_dims)
    12141    0.072    0.000    0.195    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\lib\function_base.py:564(asarray_chkfinite)
    24192    0.125    0.000    0.187    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\ma\core.py:644(get_masked_subclass)
    12129    0.030    0.000    0.182    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\core\fromnumeric.py:2322(any)
   377027    0.136    0.000    0.180    0.000 {built-in method builtins.isinstance}
    12095    0.167    0.000    0.179    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\pykalman\standard.py:186(_filter_predict)
    12096    0.110    0.000    0.175    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\scipy\linalg\lapack.py:1025(_compute_lwork)
       14    0.000    0.000    0.172    0.012 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\optuna\trial\_trial.py:76(suggest_float)
    24192    0.027    0.000    0.161    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\core\_methods.py:55(_any)
       16    0.000    0.000    0.134    0.008 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\optuna\samplers\_tpe\parzen_estimator.py:86(sample)
       16    0.001    0.000    0.133    0.008 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\optuna\samplers\_tpe\probability_distributions.py:40(sample)
       16    0.000    0.000    0.130    0.008 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\optuna\samplers\_tpe\_truncnorm.py:205(rvs)
   375023    0.130    0.000    0.130    0.000 {method 'update' of 'dict' objects}
       16    0.001    0.000    0.129    0.008 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\optuna\samplers\_tpe\_truncnorm.py:173(ppf)
   133059    0.089    0.000    0.128    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\ma\core.py:1362(getmask)
       16    0.000    0.000    0.127    0.008 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\optuna\samplers\_tpe\_truncnorm.py:180(ppf_left)
    24194    0.024    0.000    0.118    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\ma\core.py:1329(make_mask_descr)
    12153    0.019    0.000    0.116    0.000 {method 'all' of 'numpy.ndarray' objects}


