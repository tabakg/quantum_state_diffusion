Bootstrap: docker
From: tabakg/quantum_state_diffusion
IncludeCmd: yes

%runscript

    exec /usr/local/anaconda3/bin/python /code/make_quantum_trajectory.py "$@"


% post

    sudo chmod -R 777 /data 
    echo "To run, ./qsd.img --help"
