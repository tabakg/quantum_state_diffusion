Bootstrap: docker
From: tabakg/quantum_state_diffusion
IncludeCmd: yes

%runscript

    exec /usr/local/anaconda3/bin/python /code/quantum_state_diffusion.py "$@"

% post

    echo "To run, ./qsd.img [args]..."
