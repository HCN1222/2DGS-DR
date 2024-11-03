python -u train.py -s data/photo/scooters/ --eval --iterations 61000 \
        --project_name default

# exporting to mesh
cd  /home/x2ddf/HD/ning/2d-gaussian-splatting
python render.py -m /home/x2ddf/HD/ning/3DGS-DR/output/scooters/default -s /home/x2ddf/HD/ning/data/photo/scooters

# train 2dgs
python train.py -s data/photo/scooters/
python render.py -m /home/x2ddf/HD/ning/2d-gaussian-splatting/output/scooters -s /home/x2ddf/HD/ning/data/photo/scooters

#============================= ref nerf reflective =======================================================
# python -u train.py -s data/ref_nerf/ref_synthetic/ball --eval --iterations 61000 --white_background --lambda_normal 0.1 --lambda_dist0.01
# python -u train.py -s data/ref_nerf/ref_synthetic/car --eval --iterations 61000 --white_background
# python -u train.py -s data/ref_nerf/ref_synthetic/coffee --eval --iterations 61000 --white_background
# python -u train.py -s data/ref_nerf/ref_synthetic/helmet --eval --iterations 61000 --white_background
# python -u train.py -s data/ref_nerf/ref_synthetic/teapot --eval --iterations 61000 --white_background
# python -u train.py -s data/ref_nerf/ref_synthetic/toaster --eval --iterations 61000 --lambda_normal 0.01 --lambda_dist 0.1 --white_background --longer_prop_iter 24_000

# exporting to mesh
# cd  /home/x2ddf/HD/ning/2d-gaussian-splatting
# python render.py -m /home/x2ddf/HD/ning/3DGS-DR/output/ball -s /home/x2ddf/HD/ning/3DGS-DR/data/ref_nerf/ref_synthetic/ball
# python render.py -m /home/x2ddf/HD/ning/3DGS-DR/output/car -s /home/x2ddf/HD/ning/3DGS-DR/data/ref_nerf/ref_synthetic/car
# python render.py -m /home/x2ddf/HD/ning/3DGS-DR/output/coffee -s /home/x2ddf/HD/ning/3DGS-DR/data/ref_nerf/ref_synthetic/coffee
# python render.py -m /home/x2ddf/HD/ning/3DGS-DR/output/helmet -s /home/x2ddf/HD/ning/3DGS-DR/data/ref_nerf/ref_synthetic/helmet
# python render.py -m /home/x2ddf/HD/ning/3DGS-DR/output/teapot -s /home/x2ddf/HD/ning/3DGS-DR/data/ref_nerf/ref_synthetic/teapot
# python render.py -m /home/x2ddf/HD/ning/3DGS-DR/output/toaster -s /home/x2ddf/HD/ning/3DGS-DR/data/ref_nerf/ref_synthetic/toaster
# train 2dgs
# python train.py -s data/ref_nerf/ref_synthetic/ball --white_background
# python train.py -s data/ref_nerf/ref_synthetic/car --white_background
# python train.py -s data/ref_nerf/ref_synthetic/coffee --white_background 
# python train.py -s data/ref_nerf/ref_synthetic/helmet --white_background 
# python train.py -s data/ref_nerf/ref_synthetic/teapot --white_background 
# python train.py -s data/ref_nerf/ref_synthetic/toaster --white_background
# # exporting to mesh
# python render.py -m /home/x2ddf/HD/ning/2d-gaussian-splatting/output/ball -s /home/x2ddf/HD/ning/2d-gaussian-splatting/data/ref_nerf/ref_synthetic/ball
# python render.py -m /home/x2ddf/HD/ning/2d-gaussian-splatting/output/car -s /home/x2ddf/HD/ning/2d-gaussian-splatting/data/ref_nerf/ref_synthetic/car
# python render.py -m /home/x2ddf/HD/ning/2d-gaussian-splatting/output/coffee -s /home/x2ddf/HD/ning/2d-gaussian-splatting/data/ref_nerf/ref_synthetic/coffee
# python render.py -m /home/x2ddf/HD/ning/2d-gaussian-splatting/output/helmet -s /home/x2ddf/HD/ning/2d-gaussian-splatting/data/ref_nerf/ref_synthetic/helmet
# python render.py -m /home/x2ddf/HD/ning/2d-gaussian-splatting/output/teapot -s /home/x2ddf/HD/ning/2d-gaussian-splatting/data/ref_nerf/ref_synthetic/teapot
# python render.py -m /home/x2ddf/HD/ning/2d-gaussian-splatting/output/toaster -s /home/x2ddf/HD/ning/2d-gaussian-splatting/data/ref_nerf/ref_synthetic/toaster
# =================================================================================================