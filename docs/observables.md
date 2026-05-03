# Observables in the Pseudofermion Representation

このノートは、Infinite PE / pseudofermion 表示でエネルギー、比熱などをどう測るかを整理するためのメモです。実装済み API の完全な説明ではなく、今後 `measure` 系を足すときの設計ノートとして扱います。

## EDMC での現在の定義

EDMC の observable 実装は `src/EDMC/measurements.jl` にあります。1 つの Z2 ゲージ配置 `u` に対して Majorana 単一粒子エネルギーを `epsilon_k >= 0` とすると、全エネルギー estimator は

```text
E(u, beta) = -1/2 * sum_k epsilon_k * tanh(beta * epsilon_k / 2)
```

です。その beta 微分は

```text
dE(u, beta)/d beta =
    -1/4 * sum_k epsilon_k^2 * sech(beta * epsilon_k / 2)^2
```

です。

ゲージサンプル平均を `<>`、サイト数を `N` とすると、現在の EDMC の比熱 per site は

```text
C/N = beta^2 * ( (<E^2> - <E>^2) / N - <dE/d beta> / N )
```

です。コード上は `energy` を per site で保持しているため、

```text
specific_heat =
    beta^2 * (N * (energy2_per_site2 - energy_per_site^2)
              - energy_beta_derivative_per_site)
```

という形になっています。

## 有限積重みから見た observable

Infinite PE の determinant 版では、ゲージ配置 `u` の有限カットオフ重みを

```text
W_N(u, beta) = prod_{n=0}^{Ncut-1} det M_n(A(u), beta)
M_n(A, beta) = I - A / omega_n
omega_n = (2n + 1) pi / beta
```

と置きます。全体の beta 依存しない prefactor は省略します。

このとき有限カットオフでのエネルギー estimator は

```text
E_N(u, beta) = - d log W_N(u, beta) / d beta
```

です。ゲージ和を含む分配関数 `Z_N = sum_u W_N(u, beta)` に対して

```text
<E_N> = - d log Z_N / d beta
```

になります。

比熱は EDMC と同じ構造で

```text
C_N/N = beta^2 * ( Var_u(E_N) - <dE_N/d beta> ) / N
```

です。つまり determinant 版であっても、必要なのは次の 2 つです。

- 各サンプルの `E_N(u, beta)`
- 各サンプルの `dE_N(u, beta)/d beta`

実装初期段階では、pseudofermion MC で得たゲージサンプルに対して、測定時だけ dense diagonalization または determinant derivative を使うのが最も検証しやすいです。更新と測定を分離でき、pseudofermion ノイズを比熱に混ぜずに済みます。

## Pseudofermion 表示

real pseudofermion 表示では、各 Matsubara 周波数について field `phi_n` を導入します。重要なのは、`M_n` が一般に非対称なので action を `M_n^(-1)` で書かず、normal operator

```text
Q_n(A, beta) = M_n(A, beta)' * M_n(A, beta)
```

を使うことです。固定した pseudofermion field に対する action は

```text
S_pf(A, phi, beta) =
    1/2 * sum_n phi_n' * Q_n(A, beta)^(-1) * phi_n
```

です。field refresh は

```text
phi_n = M_n(A, beta)' * xi_n
xi_n ~ Normal(0, I)
```

で行います。このとき `phi_n` の covariance は `Q_n = M_n'M_n` です。

この Gaussian 積分は

```text
integral dphi_n exp(-1/2 * phi_n' * Q_n^(-1) * phi_n)
    proportional to sqrt(det Q_n)
    = |det M_n|
```

を与えます。今回の Majorana Matsubara 行列では determinant が正なので、有限積重み `prod_n det M_n` と対応します。

拡張された分布

```text
P(u, phi) proportional to exp(-S_pf(A(u), phi, beta))
```

で測るなら、エネルギー estimator は action の beta 微分として

```text
E_pf(u, phi, beta) = d S_pf(A(u), phi, beta) / d beta
```

になります。これは

```text
<E_pf>_{u,phi} = - d log Z_N / d beta
```

を満たす finite-cutoff estimator です。

## beta 微分の形

`c_n = (2n + 1) pi` と置くと

```text
omega_n = c_n / beta
M_n(beta) = I - beta * A / c_n
B_n = dM_n/d beta = -A / c_n
Q_n = M_n' * M_n
R_n = dQ_n/d beta = B_n' * M_n + M_n' * B_n
T_n = d^2Q_n/d beta^2 = 2 * B_n' * B_n
```

です。`A` が beta 非依存なので `d^2 M_n / d beta^2 = 0` ですが、`Q_n=M_n'M_n` には `T_n=2B_n'B_n` が残ります。

固定した pseudofermion field に対して

```text
dS_pf/d beta =
    -1/2 * sum_n phi_n' * Q_n^(-1) * R_n * Q_n^(-1) * phi_n
```

です。実装では

```text
x_n = Q_n \ phi_n

dS_pf/d beta =
    -1/2 * sum_n x_n' * R_n * x_n
```

と書くと、必要な solve が明確です。sparse 実装では `Q_n` を明示的に作らず、`v -> M_n' * (M_n * v)` を CG で解くのが自然です。

2 階微分は

```text
d^2 S_pf/d beta^2 =
    sum_n [
        phi_n' * Q_n^(-1) * R_n * Q_n^(-1) * R_n * Q_n^(-1) * phi_n
        - 1/2 * phi_n' * Q_n^(-1) * T_n * Q_n^(-1) * phi_n
    ]
```

です。同じ `x_n = Q_n \ phi_n` と `y_n = Q_n \ (R_n * x_n)` を使うと

```text
d^2 S_pf/d beta^2 =
    sum_n [x_n' * R_n * y_n - 1/2 * x_n' * T_n * x_n]
```

と評価できます。比熱 estimator は

```text
C_N/N =
    beta^2 / N * ( Var_{u,phi}(dS_pf/d beta)
                   - <d^2 S_pf/d beta^2>_{u,phi} )
```

になります。

この式は formal には自然ですが、実用上は pseudofermion ノイズの寄与が大きくなりやすいです。特に比熱は variance を含むため、サンプル数、field refresh の方法、自己相関の影響を強く受けます。

## 実装方針メモ

短期的には、次の 2 段階に分けるのが安全です。

1. pseudofermion MC は gauge sample を生成するために使う。
2. observable 測定では、その gauge sample ごとに EDMC 型の `E(u,beta)` と `dE/d beta`、または determinant derivative の `E_N(u,beta)` と `dE_N/d beta` を別途計算する。

この方法なら、比熱の式は既存 EDMC と同じ

```text
C/N = beta^2 * ( Var_u(E) - <dE/d beta> ) / N
```

のまま使えます。

pure pseudofermion estimator を実装する場合は、少なくとも次の値を測定結果に持たせるとよいです。

- `energy_estimator = dS_pf/d beta`
- `energy_estimator2 = (dS_pf/d beta)^2`
- `energy_beta_derivative_estimator = d^2S_pf/d beta^2`
- `specific_heat_estimator = beta^2 * (variance - mean_derivative) / nsites`
- `cutoff`
- `solver`, `operator`, `tol`, `maxiter`, `krylovdim`
- pseudofermion refresh の粒度、例えば `per_attempt` か `per_sweep`

## 注意点

- finite cutoff の observable は、厳密な EDMC observable ではなく `W_N` に対する observable。cutoff extrapolation が必要。
- pseudofermion action は `M_n` そのものではなく `Q_n=M_n'M_n` で定義する。`phi=M_n'xi` と action の covariance を必ず対応させる。
- sparse solver は `Q_n` が対称正定値であることを使い、GMRES ではなく CG を基本にする。
- 比熱は variance estimator なので、pseudofermion ノイズと Markov chain 自己相関の両方に敏感。
- まずは gauge samples に対する determinant / diagonalization 測定で EDMC と突き合わせ、その後 pure pseudofermion estimator を足すのが検証しやすい。
- `run_sparse_pseudofermion_mc` は現在、各 attempt で pseudofermion を refresh している。この設計は測定式や自己相関解析にも影響する。
