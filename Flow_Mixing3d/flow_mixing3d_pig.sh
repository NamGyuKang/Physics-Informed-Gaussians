XLA_PYTHON_CLIENT_PREALLOCATE=false python flow_mixing3d.py --model=pinn --equation=flow_mixing3d --nc=30 --nc_test=50 --seed=100 --lr=1e-2 --epochs=50000 --mlp=mlp --n_layers=2 --features=16 --out_dim=1 --pos_enc=0 --vmax=0.385 --log_iter=100 --plot_iter=10000 --mlp_dim 4 --num_gaussian 4000 --grid_range 2. --sigmas_range 0.1