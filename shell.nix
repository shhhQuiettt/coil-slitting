let
  pkgs = import <nixpkgs> {};
in pkgs.mkShell {
  packages = [
    (pkgs.python311.withPackages (python-pkgs: [
      python-pkgs.pymoo
      python-pkgs.matplotlib
      python-pkgs.numpy
      python-pkgs.seaborn
      python-pkgs.anytree
      python-pkgs.opencv4
      python-pkgs.streamlit
      python-pkgs.plotly
    ]))
  ];
}
