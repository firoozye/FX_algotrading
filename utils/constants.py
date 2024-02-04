### Constants for hyperparameter optimizations constraints

# Hyper-parameter Optimization constraints
# TODO: Read from json (already in MODEL_"CV and ALLOCATOR_CV
funding_bounds = (0.99, 1.1)  # (1-eps, 1+eps)
tcost_bounds = (0.99, 1.1)  # (1-eps, 1+eps)
risk_aversion_bounds = (1E-7, 9E-1)  # expanded due to tcost=1
st_scale_factor_bounds = (0.002, 20000)
lt_scale_factor_bounds = (0.002, 20000)
lt_posn_scale_bounds = (5E+2, 2E+6)  # st_scale_factor / risk_aversion
st_posn_scale_bounds = (5E+2, 2E+6)
width_bounds = (1E+0, 1E+6)  # 1/risk_aversion
span_bounds = (10, 5E+10)
ridge_reg_bounds = (1E-7, 1E5)
sparse_reg_bounds = (1E-2, 1E5)
