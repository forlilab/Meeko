## Notes

In `vina_params.json`, thiolate Sulfur is typed as H-bond acceptor. This differs from original
Vina typing which lacks an atom type for H-bond acceptor Sulfur: the only XS type is `S_P`.
However, modeling thiolates almost certainly requires an explicit action from the user, or a
reasonable pKa model, thus it is adequate to produce different results if a thiol is modeled
in its negative form, in which it certainly behaves as a strong H-bond acceptor.
