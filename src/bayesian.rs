use rand::Rng;
use rand_distr::{Distribution, Gamma, Normal};

/// 正規-ガンマ分布を表す構造体。
///
/// 正規-逆ガンマ分布  NG(mu, lambda, alpha, beta) = N(mu, (lambda * precision)^-1) * G(alpha, beta) を表す構造体。
/// ベイズ推定により、正規分布の平均と精度の事後分布を更新することができる。
#[derive(Debug, Clone, Copy)]
pub struct GaussInverseGamma {
    mu: f64,
    lambda: f64,
    alpha: f64,
    beta: f64,
}

#[allow(dead_code)]
impl GaussInverseGamma {
    /// 正規-ガンマ分布 NG(mu, lambda, alpha, beta) を生成する。
    ///
    /// # Arguments
    ///
    /// * `mu` - 正規分布の平均の事前分布の平均
    /// * `lambda` - サンプリングされた精度と正規分布の平均の精度との比
    /// * `alpha` - **精度 (分散の逆数)** を表すガンマ分布の形状パラメータ
    /// * `beta` - **精度 (分散の逆数)** を表すガンマ分布の尺度パラメータ
    ///
    /// # Note
    ///
    /// * `lambda` は正規分布の平均の事前分布の疑似観測回数と解釈することができる。
    /// * `beta` は正規分布の精度の事前分布の疑似観測回数の2倍と解釈することができる。
    pub fn new(mu: f64, lambda: f64, alpha: f64, beta: f64) -> Self {
        assert!(!mu.is_nan(), "mu is NaN");
        assert!(lambda > 0.0, "lambda is not positive");
        assert!(alpha > 0.0, "alpha is not positive");
        assert!(beta > 0.0, "beta is not positive");

        Self {
            mu,
            lambda,
            alpha,
            beta,
        }
    }

    /// 対象とする正規分布からの疑似観測値から正規-ガンマ分布 NG(mu, lambda, alpha, beta) を生成する。
    ///
    /// # Arguments
    ///
    /// - `mean` - 疑似観測値の期待値
    /// - `std_dev` - 疑似観測値の標準偏差
    /// - `pseudo_observation_count` - 疑似観測回数
    pub fn from_psuedo_observation(
        mean: f64,
        std_dev: f64,
        pseudo_observation_count: usize,
    ) -> Self {
        assert!(std_dev > 0.0, "expected_std_dev is not positive");
        assert!(
            pseudo_observation_count > 0,
            "pseudo_observation_count is not positive"
        );

        let expected_variance = std_dev * std_dev;
        let expected_precision = 1.0 / expected_variance;

        let mu = mean;
        let lambda = pseudo_observation_count as f64;
        let alpha = (pseudo_observation_count * 2) as f64;

        // 精度の期待値E[p] = alpha / beta より、 beta = alpha / E[p]
        let beta = alpha / expected_precision;

        Self::new(mu, lambda, alpha, beta)
    }

    /// 観測値xを元にしてベイズ更新を行う。
    pub fn update(&mut self, x: f64) {
        let mu = (x + self.lambda * self.mu) / (self.lambda + 1.0);
        let lambda = self.lambda + 1.0;
        let alpha = self.alpha + 0.5;
        let dev2 = (x - self.mu) * (x - self.mu);
        let beta = self.beta + 0.5 * (self.lambda * dev2) / (self.lambda + 1.0);

        self.mu = mu;
        self.lambda = lambda;
        self.alpha = alpha;
        self.beta = beta;
    }

    /// (平均, 標準偏差) の期待値を取得する。
    pub fn expected(&self) -> (f64, f64) {
        let expected_precision = self.alpha / self.beta;
        let expected_variance = 1.0 / expected_precision;
        let expected_std_dev = expected_variance.sqrt();
        (self.mu, expected_std_dev)
    }

    pub fn get_pseudo_observation_count(&self) -> f64 {
        self.lambda
    }

    pub fn set_pseudo_observation_count(&mut self, pseudo_observation_count: f64) {
        self.lambda = pseudo_observation_count;
        self.alpha = pseudo_observation_count * 0.5;
    }
}

impl Distribution<(f64, f64)> for GaussInverseGamma {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> (f64, f64) {
        // ガンマ分布から精度をサンプリング
        let precision = rng.sample(Gamma::new(self.alpha, 1.0 / self.beta).unwrap());
        let std_dev = 1.0 / precision.sqrt();

        // 正規分布から平均をサンプリング
        let std_dev_mean = 1.0 / (precision * self.lambda).sqrt();
        let mean = rng.sample(Normal::new(self.mu, std_dev_mean).unwrap());

        (mean, std_dev)
    }
}
