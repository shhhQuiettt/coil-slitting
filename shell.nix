let
  pkgs = import <nixpkgs> {};
in pkgs.mkShell {
  packages = [
    (pkgs.python312.withPackages (python-pkgs: [
      python-pkgs.pymoo
      python-pkgs.matplotlib
      python-pkgs.numpy
      python-pkgs.seaborn
      python-pkgs.anytree
    ]))
  ];
}
