runai submit --name vqvae-smm \
 -i aicregistry:5000/nglazman:vqvae-base \
 --node-type A100 \
 --run-as-user \
 --gpu 1 \
 --cpu 16 \
 --cpu-limit 32 \
 --memory 64G --memory-limit 128G --project nglazman \
 -v /nfs:/nfs --large-shm --command -- bash /nfs/home/nglazman/vqvae-smm/docker/run_training.sh
