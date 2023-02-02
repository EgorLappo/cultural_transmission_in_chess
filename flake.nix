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
        ];
      };

      python-env = pkgs.python39.withPackages (ps: with ps; [ 
        chess
        tqdm
        numpy
        pandas
        matplotlib
        seaborn
        unidecode
      ]);
    in {
      devShells.default = with pkgs; mkShell {
        name = "R";
        buildInputs = [
          R-env python-env cmdstan
        ];

        shellHook = ''
          mkdir -p "$(pwd)/_libs"
          export R_LIBS_USER="$(pwd)/_libs"
          export PYTHONPATH="${python-env}/bin/python"
        '';
      };
    });
}
