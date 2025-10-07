DETAILED OPTUNA PROFILING REPORT
============================================================

OPTIMIZATION ANALYSIS
-------------------------
Total trials: 50
Total time: 18.1976s
Average trial time: 0.3640s
Median trial time: 0.3352s
Min trial time: 0.0878s
Max trial time: 2.7981s
Std trial time: 0.3585s
Optimization efficiency: 63.18%

SLOWEST TRIALS
--------------------
1. Trial 0: 2.7981s
2. Trial 45: 0.4274s
3. Trial 30: 0.4216s
4. Trial 31: 0.4216s
5. Trial 44: 0.4139s

FASTEST TRIALS
--------------------
1. Trial 9: 0.1493s
2. Trial 8: 0.1453s
3. Trial 7: 0.1080s
4. Trial 6: 0.1056s
5. Trial 5: 0.0878s

FUNCTION HOTSPOTS
--------------------
SLOWEST FUNCTIONS
--------------------
DETAILED CPROFILE STATS
-------------------------
         118583 function calls (118580 primitive calls) in 0.412 seconds

   Ordered by: cumulative time
   List reduced from 578 to 50 due to restriction <50>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.412    0.412 C:\visual projects\backtrader\src\optimization\fast_optimizer.py:301(objective)
       16    0.000    0.000    0.255    0.016 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\optuna\trial\_trial.py:611(_suggest)
       16    0.000    0.000    0.252    0.016 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\optuna\samplers\_tpe\sampler.py:395(sample_independent)
       16    0.001    0.000    0.251    0.016 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\optuna\samplers\_tpe\sampler.py:437(_sample)
       14    0.000    0.000    0.217    0.015 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\optuna\trial\_trial.py:76(suggest_float)
       16    0.000    0.000    0.152    0.010 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\optuna\samplers\_tpe\parzen_estimator.py:86(sample)
        1    0.008    0.008    0.152    0.152 C:\visual projects\backtrader\src\strategies\turbo_mean_reversion_strategy.py:569(turbo_process_dataset)
       16    0.001    0.000    0.152    0.009 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\optuna\samplers\_tpe\probability_distributions.py:40(sample)
       16    0.000    0.000    0.148    0.009 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\optuna\samplers\_tpe\_truncnorm.py:205(rvs)
       16    0.001    0.000    0.147    0.009 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\optuna\samplers\_tpe\_truncnorm.py:173(ppf)
       16    0.000    0.000    0.145    0.009 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\optuna\samplers\_tpe\_truncnorm.py:180(ppf_left)
       16    0.001    0.000    0.126    0.008 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\optuna\samplers\_tpe\_truncnorm.py:169(_ndtri_exp)
      384    0.001    0.000    0.124    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\optuna\samplers\_tpe\_truncnorm.py:164(_ndtri_exp_single)
      384    0.046    0.000    0.124    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\optuna\samplers\_tpe\_truncnorm.py:151(_bisect)
    20549    0.045    0.000    0.084    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\optuna\samplers\_tpe\_truncnorm.py:77(_log_ndtr_single)
        1    0.057    0.057    0.080    0.080 C:\visual projects\backtrader\src\strategies\turbo_mean_reversion_strategy.py:77(vectorized_ou_half_life)
       52    0.003    0.000    0.064    0.001 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\optuna\samplers\_tpe\_truncnorm.py:112(_log_gauss_mass)
       32    0.000    0.000    0.062    0.002 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\optuna\samplers\_tpe\parzen_estimator.py:90(log_pdf)
       32    0.004    0.000    0.062    0.002 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\optuna\samplers\_tpe\probability_distributions.py:80(log_pdf)
        1    0.000    0.000    0.058    0.058 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\sklearn\mixture\_base.py:155(fit)
      2/1    0.000    0.000    0.058    0.058 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\sklearn\base.py:1346(wrapper)
        1    0.000    0.000    0.057    0.057 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\sklearn\mixture\_base.py:185(fit_predict)
       51    0.001    0.000    0.052    0.001 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\optuna\samplers\_tpe\_truncnorm.py:127(mass_case_central)
      102    0.002    0.000    0.052    0.001 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\optuna\samplers\_tpe\_truncnorm.py:72(_ndtr)
      102    0.009    0.000    0.050    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\optuna\samplers\_tpe\_erf.py:108(erf)
       28    0.003    0.000    0.040    0.001 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\optuna\samplers\_tpe\_truncnorm.py:218(logpdf)
        2    0.000    0.000    0.039    0.019 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\optuna\trial\_trial.py:238(suggest_int)
        5    0.000    0.000    0.033    0.007 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\sklearn\mixture\_base.py:296(_e_step)
        5    0.001    0.000    0.032    0.006 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\sklearn\mixture\_base.py:513(_estimate_log_prob_resp)
      568    0.006    0.000    0.031    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\polynomial\_polybase.py:510(__call__)
    20409    0.021    0.000    0.029    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\optuna\samplers\_tpe\_truncnorm.py:58(_ndtr_single)
        5    0.003    0.001    0.026    0.005 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\scipy\special\_logsumexp.py:15(logsumexp)
      392    0.026    0.000    0.026    0.000 {method 'reduce' of 'numpy.ufunc' objects}
      568    0.022    0.000    0.024    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\polynomial\polynomial.py:664(polyval)
      200    0.001    0.000    0.022    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\core\fromnumeric.py:71(_wrapreduction)
        5    0.008    0.002    0.020    0.004 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\scipy\special\_logsumexp.py:205(_logsumexp)
      108    0.001    0.000    0.017    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\numpy\core\fromnumeric.py:2177(sum)
       32    0.001    0.000    0.015    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\optuna\samplers\_tpe\parzen_estimator.py:44(__init__)
        4    0.001    0.000    0.013    0.003 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\sklearn\mixture\_gaussian_mixture.py:819(_m_step)
       82    0.003    0.000    0.013    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\optuna\samplers\_tpe\_erf.py:138(calc_case_med1)
      101    0.001    0.000    0.012    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\optuna\samplers\_tpe\_erf.py:124(calc_case_small1)
       32    0.004    0.000    0.011    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\optuna\samplers\_tpe\sampler.py:425(_get_internal_repr)
       32    0.003    0.000    0.010    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\optuna\samplers\_tpe\_truncnorm.py:104(_log_ndtr)
    20809    0.010    0.000    0.010    0.000 {built-in method math.log}
        1    0.000    0.000    0.010    0.010 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\sklearn\mixture\_gaussian_mixture.py:775(_initialize_parameters)
        1    0.000    0.000    0.009    0.009 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\sklearn\mixture\_base.py:98(_initialize_parameters)
       64    0.002    0.000    0.009    0.000 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\optuna\samplers\_tpe\_erf.py:153(calc_case_med2)
        5    0.006    0.001    0.009    0.002 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\sklearn\mixture\_gaussian_mixture.py:258(_estimate_gaussian_parameters)
        8    0.000    0.000    0.009    0.001 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\optuna\samplers\_tpe\_truncnorm.py:121(mass_case_left)
        1    0.001    0.001    0.008    0.008 C:\Users\Георгий\AppData\Local\Programs\Python\Python311\Lib\site-packages\sklearn\cluster\_kmeans.py:1426(fit)


