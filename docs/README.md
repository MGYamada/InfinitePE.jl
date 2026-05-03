# InfinitePE.jl Notes

このディレクトリは、本格的なユーザー向けドキュメントではなく、これまでの実装と理論の対応を残すためのノートです。

追加ノート:

- [Observables in the Pseudofermion Representation](observables.md)

## 目的

`InfinitePE.jl` は、Kitaev 型スピン模型に対して、従来の EDMC / Full ED ベースラインと比較しながら、Majorana 自由フェルミオンの Boltzmann 重みを Matsubara 周波数の有限カットオフ積で扱うための実験的な実装です。

現時点の主な関心は次の 3 点です。

- 小さい格子で Full ED と EDMC の規約を確認する。
- Z2 ゲージ配置ごとの Majorana ハミルトニアンから、自由エネルギー差または有限積重みを計算する。
- determinant 版と pseudofermion 版の Infinite PE 更新を比較できる形にする。

## 格子と Kitaev 入力

格子生成は `src/lattices.jl` にあります。

- `generate_honeycomb(Lx, Ly, bc)` は 2 副格子の honeycomb 格子を作る。
- `generate_hyperhoneycomb(Lx, Ly, Lz, bc)` は 4 副格子の hyperhoneycomb 格子を作る。
- 境界条件は `TypeI` と `TypeII`。現状、`TypeII` は honeycomb 専用。
- 各 bond は `BondEdge(src, dst, bond, wrapped)` として保持し、`bond` は `:x`, `:y`, `:z` のいずれか。

EDMC 側では `EDMC.extract_kitaev_bonds` と `EDMC.lattice_to_edmc` が格子を `KitaevHamiltonianInput` に変換します。bond の順序は安定化されており、Z2 ゲージ場 `u_ij = +/-1` は bond index に沿って保持されます。

## Majorana 規約

Majorana 単一粒子ハミルトニアンは

```text
H = i A
```

と置きます。ここで `H` は純虚数 Hermitian 行列、`A` は実反対称行列です。

`build_majorana_matrix(input)` は EDMC 入力から `A` を作ります。各 Kitaev bond の寄与は

```text
A_ij += 2 J_gamma u_ij
A_ji -= 2 J_gamma u_ij
```

です。この `2J_gamma u_ij` のスケールは、EDMC の自由エネルギー差と Infinite PE の Boltzmann 重み差が一致するように選んでいます。

テストでは

```text
build_majorana_matrix(input) ≈ 2 * majorana_matrix(EDMC.build_hamiltonian(input))
```

を確認しています。

## Infinite Product Expansion

有限温度 `beta` に対して、正の fermionic Matsubara 周波数を

```text
omega_n = (2n + 1) pi / beta,  n = 0, 1, ...
```

とします。実装では zero-based index `n` を使います。

Majorana 行列 `A` から Matsubara 行列

```text
M_n(A) = I - A / omega_n
```

を作り、有限カットオフ `N_cut` の determinant 重みを

```text
log W_N(A) = sum_{n=0}^{N_cut-1} log det M_n(A)
```

で近似します。全体の定数 prefactor は省略しています。Metropolis 更新では同じサイズの行列間の差だけを使うため、この定数はキャンセルします。

実装入口は `src/infinite_product_expansion.jl` です。

- `matsubara_frequency`
- `matsubara_matrix`
- `logdet_matsubara_dense`
- `log_weight_infinite_product`
- `delta_log_weight_infinite_product`
- `acceptance_probability_logweight`

`log_weight` / `delta_log_weight` / `acceptance_probability` は短い alias として用意されています。

## EDMC との対応

EDMC では gauge flip の受理率を、自由エネルギー差 `Delta F` から

```text
min(1, exp(-beta * Delta F))
```

として評価します。

Infinite PE の determinant 版では

```text
Delta log W_N = log W_N(after) - log W_N(before)
min(1, exp(Delta log W_N))
```

を使います。カットオフを大きくすると、テスト上は

```text
Delta log W_N -> -beta * Delta F
```

に収束します。`test/infinite_product_expansion.jl` では `cutoff=100` で EDMC の自由エネルギー差と一致することを確認しています。

## Real Pseudofermion 表現

determinant を直接評価する代わりに、real pseudofermion field を導入する実装もあります。

各 Matsubara 周波数について標準正規乱数 `xi_n` を引き、

```text
phi_n = M_n(A)' xi_n
```

として pseudofermion field を refresh します。固定した field に対する action は

```text
Q_n(A) = M_n(A)' * M_n(A)
S_pf(A, phi) = 1/2 * sum_n phi_n' * Q_n(A)^(-1) * phi_n
```

です。`M_n` は一般に非対称なので、action は `M_n^(-1)` ではなく
positive definite な normal operator `M_n'M_n` で定義します。この形にすると

```text
integral dphi exp(-S_pf) proportional to prod_n det M_n(A)
```

となり、有限積 determinant 重みと対応します。更新時には同じ `phi` を before / after で評価し、

```text
Delta S_pf = S_pf(after, phi) - S_pf(before, phi)
min(1, exp(-Delta S_pf))
```

で受理率を計算します。

dense 実装の入口は次です。

- `refresh_real_pseudofermions`
- `pseudofermion_action`
- `delta_pseudofermion_action`
- `acceptance_probability_pseudofermion`

小さい系では、determinant MC と pseudofermion MC の平均 `log W` が exact enumeration と同程度に一致することをテストしています。

## Sparse 実装

`src/sparse.jl` には、より大きい格子へ向けた sparse / matrix-free 実装があります。

- `build_sparse_majorana_matrix` は sparse な `A` を作る。
- `sparse_matsubara_mul` は `M_n v = v - A v / omega_n` を直接適用する。
- `sparse_matsubara_transpose_mul` は `M_n' v = v + A v / omega_n` を適用する。
- `sparse_pseudofermion_action` は `solver=:cg` または `solver=:direct` で `M_n'M_n` の線形方程式を解く。
- `run_sparse_pseudofermion_mc` は pseudofermion を毎 attempt refresh する簡単な Z2 bond-flip MC。

CG の場合、`operator=:matrix_free` なら `M_n'M_n` を明示的に構成せず、`M_n` と `M_n'` の sparse matvec で normal operator を適用します。`operator=:matrix` なら sparse Matsubara 行列を作って `M_n'M_n` を構成します。

この部分は実験的です。収束しない場合は `tol`, `maxiter`, `krylovdim`, `solver`, `operator` を見直す必要があります。

## 比較用ベースライン

`src/EDMC` は Majorana 自由フェルミオンの EDMC baseline です。

- gauge 初期化と bond flip
- Hamiltonian 構築
- diagonalization
- free energy / internal energy / specific heat
- 温度 scan
- comparison table

`src/FullED` は小さいスピン Hilbert 空間での Full ED baseline です。

- full spin Hamiltonian 構築
- partition function / free energy / entropy / specific heat
- 温度 scan
- comparison table

比較行は method-agnostic な `NamedTuple` を返す方針です。EDMC と FullED で field が完全に同一ではないため、プロットや CSV 化では必要な列を選んで使います。

## スクリプト

`scripts/` には PRL 113, 197205 の Fig. 5 風の比較を意識した軽量 baseline があります。

```sh
julia --project=. scripts/edmc_prl113197205_baseline.jl
julia --project=. scripts/fulled_prl113197205_baseline.jl
julia --project=. scripts/plot_prl113197205_fig5.jl
```

デフォルトは小さい honeycomb 格子と短い MC run なので、定量的な本番結果ではなく、energy curve や specific heat peak の定性的な確認用です。

## 現時点の確認事項

テストで確認している主な内容は次です。

- lattice から EDMC bondset への変換が安定している。
- EDMC の Hamiltonian は Hermitian で、Majorana energy convention が FullED と小さい系で合う。
- Infinite PE の 2x2 解析例で Matsubara determinant が期待式に一致する。
- Infinite PE の determinant 重み差が EDMC の `-beta * Delta F` に cutoff とともに収束する。
- dense pseudofermion action は `phi = M' xi` のとき `1/2 * xi'xi` に一致する。
- sparse pseudofermion action は dense 実装と小さい系で一致する。

## 未整理・今後のメモ

- Infinite PE の「本命」ドライバはまだ薄い。現状は determinant / pseudofermion の部品と sparse MC が中心。
- カットオフ依存性の系統的な scan はまだ整理していない。
- pseudofermion refresh を attempt ごとに行っているが、性能や統計効率の観点では更新設計を再検討する余地がある。
- sparse CG の収束性は模型・温度・カットオフ・格子サイズに依存するため、失敗時の診断ログや fallback 方針が必要。
- comparison table の field set は EDMC / FullED / Infinite PE でさらに揃える余地がある。
- 本格的な Documenter.jl 化は未着手。このノートは、その前段階の実装メモとして扱う。
