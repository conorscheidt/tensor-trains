{
  description = "Tensor trains vs. Monte Carlo: experiments";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "aarch64-darwin" "x86_64-darwin" "aarch64-linux" "x86_64-linux" ];
      forAll = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      # Nix provides uv; uv.lock pins everything else.
      devShells = forAll (pkgs: {
        default = pkgs.mkShell {
          packages = [ pkgs.uv ];
          shellHook = "export UV_PYTHON_DOWNLOADS=automatic";
        };
      });
    };
}
