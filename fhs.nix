{ pkgs ? import <nixpkgs> {} }:
(pkgs.buildFHSUserEnv {
  name = "YOLO for Autonomous Driving";
  targetPkgs = pkgs: with pkgs; [
    # Defaults for cuda
    git
    gitRepo
    gnupg
    autoconf
    curl
    procps
    gnumake
    util-linux
    m4
    gperf
    unzip
    cudatoolkit
    linuxPackages.nvidia_x11
    libGLU libGL
    xorg.libXi xorg.libXmu freeglut
    xorg.libXext xorg.libX11 xorg.libXv xorg.libXrandr zlib 
    ncurses5
    stdenv.cc
    binutils
    #CV2
    glib
    #IDE
    jetbrains.pycharm-professional
    # Mongodb for Fiftyone
    mongodb-ce
  ];
  multiPkgs = pkgs: with pkgs; [ zlib ];
  profile = ''
    #Default cuda
    export CUDA_PATH=${pkgs.cudatoolkit}
    # export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib
    export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
    export EXTRA_CCFLAGS="-I/usr/include"
    # CV2
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${pkgs.glib}/lib"
    #IDE
    export PYDEVD_USE_CYTHON=NO
    # Create a data directory if it doesn't exist
    mkdir -p /tmp/mongodb-data
    echo "Starting MongoDB on port 27017 with dbpath /tmp/mongodb-data..."
    # Start MongoDB in the background
    mongod --dbpath /tmp/mongodb-data --fork --port 27017 --logpath /tmp/mongodb.log
  '';
}).env
