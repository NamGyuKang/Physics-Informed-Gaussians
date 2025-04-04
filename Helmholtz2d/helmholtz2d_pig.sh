# python main.py --network base --pde helmholtz_2d --a1 4.0 --a2 1.0 --in-dim 2 --out-dim 1 --mlp-dim 4 --num-train 10000 --num-init 10000 --num-test 250 --random-f --hidden-dim 16 --num-layers 2 --max-iter 60001 --f-scale 0.0001 --seed 100 --num-gaussians 800 --sigma-init 0.1 --optim adam --lr 0.01 --tag helmholtz2d 

for sd in 100 200 300 400 500
do
    CUDA_VISIBLE_DEVICES=2 python main.py --network base --pde helmholtz_2d --a1 4.0 --a2 1.0 --in-dim 2 --out-dim 1 --mlp-dim 4 --num-train 10000 --num-init 10000 --num-test 250 --random-f --hidden-dim 16 --num-layers 2 --max-iter 2001 --f-scale 0.0001 --seed $sd --num-gaussians 800 --sigma-init 0.1 --tag helmholtz2d_lbfgs_$sd 
done