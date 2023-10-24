## notes
# without the following lines, dplyr does not work 
#     mkdir -p "$(pwd)/_libs"
#     export R_LIBS_USER="$(pwd)/_libs"
## 


{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }: 
    
  flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = import nixpkgs { inherit system; };

      cmdstanr = pkgs.rPackages.buildRPackage {
        name = "cmdstanr";
        src = pkgs.fetchFromGitHub {
          owner = "stan-dev";
          repo = "cmdstanr";
          rev="b5d3a77c94e48cf84546c76f613a48282a9e4543";
          sha256="0w91ixbycz578sddwsvml8gvdc7pg4zfdxk5yrn5wii4r1vmzxq0";
          };
        propagatedBuildInputs = with pkgs.rPackages; [
            data_table jsonlite checkmate posterior processx R6 withr 
          ];
        };
      R-env = pkgs.rWrapper.override {
        packages = with pkgs.rPackages; [
          tidyverse 
          cmdstanr
          posterior
          bayesplot
          scales
          ggh4x
          patchwork
        ];
      };
      Rscriptpath = "${R-env}/bin/Rscript";
      
      dontTestPackage = drv: drv.overridePythonAttrs (old: { doCheck = false; });
      python-env = pkgs.python311.withPackages (ps: with ps; [ 
        pip
        chess
        tqdm
        numpy
        pandas
        matplotlib
        (dontTestPackage seaborn) # tests fail on darwin due to different numerical results on intel vs ARM
        unidecode
      ]);

      cmdstan = pkgs.cmdstan;
      cmdstanpath = "${cmdstan}/opt/cmdstan";
    in rec {
      devShells.default = with pkgs; mkShell {
        name = "chess_model";
        buildInputs = [
          R-env python-env cmdstan ouch
        ];

        CMDSTAN = cmdstanpath;
        shellHook = ''
          mkdir -p "$(pwd)/_libs"
          export R_LIBS_USER="$(pwd)/_libs"
          export PYTHONPATH="${python-env}/bin/python"
          alias Rs="${Rscriptpath} --vanilla"
        '';
      };
    });
}
