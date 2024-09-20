{
  description = "A Python development environment with poetry";

  inputs = {
    nixpkgs.url = "nixpkgs";
    easy-sync.url = "github:luochen1990/easy-sync/master";
  };

  outputs = { self, nixpkgs, easy-sync }:
  let
    supportedSystems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
    eachSystem = f: nixpkgs.lib.genAttrs supportedSystems (system: f rec {
      inherit system;
      pkgs = import nixpkgs { inherit system; };
      python = pkgs.python3; # use pkgs.python3.withPackages (p: []) if you need more python packages in nixpkgs
    });
  in
  {

    packages = eachSystem ({pkgs, system, ...}: rec {
      default = ai-powered;
      ai-powered = let
        pyproject = pkgs.lib.importTOML ./pyproject.toml;
        meta = pyproject.tool.poetry;
        pypkgs = pkgs.python3Packages;
      in
      pypkgs.buildPythonPackage {
        pname = meta.name;
        version = meta.version;
        pyproject = true;
        src = ./.;
        buildInputs = [ pypkgs.poetry-core ];
        dependencies = [
          easy-sync.packages.${system}.easy-sync
          pypkgs.msgspec
          pypkgs.openai
        ];
        doCheck = true;
        meta = with pkgs.lib; {
          description = meta.description;
          homepage = meta.repository;
          license = licenses.asl20;
          maintainers = with maintainers; [ luochen1990 ];
        };
      };
    });

    devShells = eachSystem ({pkgs, python, ...}: rec {
      default = poetry;

      poetry = pkgs.mkShell {
        buildInputs = [
          python
          pkgs.poetry
        ];

        shellHook = ''
          export PATH=$(poetry env info --path)/bin:$PATH
        '';
      };
    });

    apps = eachSystem ({system, pkgs, ...}: {
      default = {
        type = "app";
        program = "${pkgs.writeShellScript "funix-app" ''
          source ${self.devShells.${system}.default.shellHook}
          funix .
        ''}";
      };
    });

  };
}
