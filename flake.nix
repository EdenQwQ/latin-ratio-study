{
  description = "A Python API for various tools I use at work.";
  inputs = {
    nixpkgs.url = github:NixOS/nixpkgs/nixos-23.11;
    flake-utils = {
      url = github:numtide/flake-utils;
    };
    mach-nix = {
      url = "mach-nix/3.5.0";
    };
  };

  outputs = { self, nixpkgs, flake-utils, mach-nix, ... }@inputs:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        machNix = mach-nix.lib."${system}";
        devEnvironment = machNix.mkPython {
          requirements = builtins.readFile ./requirements.txt;
          _.pytest-cov.propagatedBuildInputs.mod = pySelf: self: oldVal: oldVal ++ [ pySelf.tomli ];
        };
      in
      {
        devShell = pkgs.mkShell {
          buildInputs = [
            devEnvironment
            pkgs.gnumake
          ];
          shellHook = ''
            export DJANGO_SETTINGS_MODULE=network_inventory.settings.local
          '';
        };
        packages.venv = devEnvironment;
      });
}
