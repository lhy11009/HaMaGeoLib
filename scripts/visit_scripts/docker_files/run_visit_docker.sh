#!/bin/bash
container_id="$1"
dir_name=`dirname $(pwd)`
case_name=`basename "${dir_name}"`
echo "${dir_name}"
docker cp "../../${case_name}" ${container_id}:/home/visit/data  # copy data into the container
docker exec "${container_id}" bash "/home/visit/data/${case_name}/visit_scripts/run_docker.sh" # execute the plotting
docker cp "${container_id}":/home/visit/data/${case_name}/img .  # copy the generated image
docker exec -u root "${container_id}" rm -rf /home/visit/data/${case_name}  # remove data in docker
