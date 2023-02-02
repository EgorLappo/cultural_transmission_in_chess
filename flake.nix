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
        ];
      };
      Rscriptpath = "${R-env}/bin/Rscript";
      
      dontTestPackage = drv: drv.overridePythonAttrs (old: { doCheck = false; });
      python-env = pkgs.python3.withPackages (ps: with ps; [ 
        chess
        tqdm
        numpy
        pandas
        matplotlib
        (dontTestPackage seaborn) # tests fail due to different numerical results on intel vs ARM
        unidecode
      ]);

      cmdstan = pkgs.cmdstan;
      cmdstanpath = "${cmdstan}/opt/cmdstan";
    in rec {
      devShells.default = with pkgs; mkShell {
        name = "R";
        buildInputs = [
          R-env python-env cmdstan
        ];

        CMDSTAN = cmdstanpath;
        shellHook = ''
          mkdir -p "$(pwd)/_libs"
          export R_LIBS_USER="$(pwd)/_libs"
          export PYTHONPATH="${python-env}/bin/python"
        '';
      };

      packages = {
        makeFigures = pkgs.writeScriptBin "makeFigures" ''
          mkdir -p "$(pwd)/_libs"
          export R_LIBS_USER="$(pwd)/_libs"
          cd make_figures
          ${python-env}/bin/python figure_2.py
          ${python-env}/bin/python figure_3.py
          ${python-env}/bin/python figure_4.py
          ${Rscriptpath} --vanilla figure_5.R 
          ${Rscriptpath} --vanilla figure_6.R
        '';
      
        runModel = pkgs.writeScriptBin "runModel" ''
          mkdir -p "$(pwd)/_libs"
          export R_LIBS_USER="$(pwd)/_libs"
          cd model
          ${python-env}/bin/python prepare_data.py
          CMDSTAN=${cmdstanpath} ${Rscriptpath} --vanilla run_stan.R
          cd ..
        '';

        generateTables = pkgs.writeScriptBin "generateTables" ''
          cd data_processing
          ${python-env}/bin/python clean.py
          ${python-env}/bin/python generate_tables.py
          cd ..
        '';

        parsePGN = pkgs.writeScriptBin "parsePGN" ''
          cd data_processing
          ${python-env}/bin/python parse.py
          cd ..
        '';
      };

      defaultPackage = packages.makeFigures;
    });
}
