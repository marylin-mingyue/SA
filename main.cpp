#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace opt {

struct EvalResult {
  double value = 0.0;
  std::vector<double> grad;
};

double dot(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size()) {
    throw std::invalid_argument("dot(): vector sizes do not match.");
  }
  double s = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    s += a[i] * b[i];
  }
  return s;
}

double norm(const std::vector<double>& v) { return std::sqrt(dot(v, v)); }

std::vector<double> add(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size()) {
    throw std::invalid_argument("add(): vector sizes do not match.");
  }
  std::vector<double> out(a.size(), 0.0);
  for (size_t i = 0; i < a.size(); ++i) {
    out[i] = a[i] + b[i];
  }
  return out;
}

std::vector<double> subtract(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size()) {
    throw std::invalid_argument("subtract(): vector sizes do not match.");
  }
  std::vector<double> out(a.size(), 0.0);
  for (size_t i = 0; i < a.size(); ++i) {
    out[i] = a[i] - b[i];
  }
  return out;
}

std::vector<double> scale(const std::vector<double>& v, double alpha) {
  std::vector<double> out(v.size(), 0.0);
  for (size_t i = 0; i < v.size(); ++i) {
    out[i] = alpha * v[i];
  }
  return out;
}

std::vector<double> add_scaled(const std::vector<double>& x, const std::vector<double>& p, double alpha) {
  if (x.size() != p.size()) {
    throw std::invalid_argument("add_scaled(): vector sizes do not match.");
  }
  std::vector<double> out(x.size(), 0.0);
  for (size_t i = 0; i < x.size(); ++i) {
    out[i] = x[i] + alpha * p[i];
  }
  return out;
}

void print_vector(const std::vector<double>& x) {
  std::cout << "[";
  for (size_t i = 0; i < x.size(); ++i) {
    std::cout << x[i];
    if (i + 1 != x.size()) {
      std::cout << ", ";
    }
  }
  std::cout << "]";
}

// -------- Forward-mode AutoDiff (dual number with full Jacobian row) --------
class Dual {
 public:
  Dual() = default;
  explicit Dual(double value) : value_(value) {}
  Dual(double value, size_t dim, size_t active_idx) : value_(value), grad_(dim, 0.0) {
    if (active_idx < dim) {
      grad_[active_idx] = 1.0;
    }
  }

  double value() const { return value_; }
  const std::vector<double>& grad() const { return grad_; }
  size_t dim() const { return grad_.size(); }

  friend Dual operator+(const Dual& a, const Dual& b) {
    const size_t n = std::max(a.dim(), b.dim());
    Dual out;
    out.value_ = a.value_ + b.value_;
    out.grad_.assign(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
      const double ga = (i < a.dim()) ? a.grad_[i] : 0.0;
      const double gb = (i < b.dim()) ? b.grad_[i] : 0.0;
      out.grad_[i] = ga + gb;
    }
    return out;
  }

  friend Dual operator-(const Dual& a, const Dual& b) {
    const size_t n = std::max(a.dim(), b.dim());
    Dual out;
    out.value_ = a.value_ - b.value_;
    out.grad_.assign(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
      const double ga = (i < a.dim()) ? a.grad_[i] : 0.0;
      const double gb = (i < b.dim()) ? b.grad_[i] : 0.0;
      out.grad_[i] = ga - gb;
    }
    return out;
  }

  friend Dual operator*(const Dual& a, const Dual& b) {
    const size_t n = std::max(a.dim(), b.dim());
    Dual out;
    out.value_ = a.value_ * b.value_;
    out.grad_.assign(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
      const double ga = (i < a.dim()) ? a.grad_[i] : 0.0;
      const double gb = (i < b.dim()) ? b.grad_[i] : 0.0;
      out.grad_[i] = ga * b.value_ + gb * a.value_;
    }
    return out;
  }

  friend Dual operator/(const Dual& a, const Dual& b) {
    const size_t n = std::max(a.dim(), b.dim());
    if (std::abs(b.value_) < 1e-15) {
      throw std::runtime_error("Dual division by near-zero.");
    }
    Dual out;
    out.value_ = a.value_ / b.value_;
    out.grad_.assign(n, 0.0);
    const double inv = 1.0 / (b.value_ * b.value_);
    for (size_t i = 0; i < n; ++i) {
      const double ga = (i < a.dim()) ? a.grad_[i] : 0.0;
      const double gb = (i < b.dim()) ? b.grad_[i] : 0.0;
      out.grad_[i] = (ga * b.value_ - gb * a.value_) * inv;
    }
    return out;
  }

  friend Dual operator-(const Dual& a) {
    Dual out;
    out.value_ = -a.value_;
    out.grad_.resize(a.grad_.size(), 0.0);
    for (size_t i = 0; i < a.grad_.size(); ++i) {
      out.grad_[i] = -a.grad_[i];
    }
    return out;
  }

  friend Dual sin(const Dual& a) {
    Dual out;
    out.value_ = std::sin(a.value_);
    out.grad_.resize(a.grad_.size(), 0.0);
    const double c = std::cos(a.value_);
    for (size_t i = 0; i < a.grad_.size(); ++i) {
      out.grad_[i] = c * a.grad_[i];
    }
    return out;
  }

  friend Dual cos(const Dual& a) {
    Dual out;
    out.value_ = std::cos(a.value_);
    out.grad_.resize(a.grad_.size(), 0.0);
    const double m = -std::sin(a.value_);
    for (size_t i = 0; i < a.grad_.size(); ++i) {
      out.grad_[i] = m * a.grad_[i];
    }
    return out;
  }

  friend Dual exp(const Dual& a) {
    Dual out;
    out.value_ = std::exp(a.value_);
    out.grad_.resize(a.grad_.size(), 0.0);
    for (size_t i = 0; i < a.grad_.size(); ++i) {
      out.grad_[i] = out.value_ * a.grad_[i];
    }
    return out;
  }

  friend Dual log(const Dual& a) {
    if (a.value_ <= 0.0) {
      throw std::runtime_error("Dual log() domain error.");
    }
    Dual out;
    out.value_ = std::log(a.value_);
    out.grad_.resize(a.grad_.size(), 0.0);
    for (size_t i = 0; i < a.grad_.size(); ++i) {
      out.grad_[i] = a.grad_[i] / a.value_;
    }
    return out;
  }

  friend Dual pow(const Dual& a, double p) {
    if (a.value_ < 0.0 && std::abs(std::round(p) - p) > 1e-12) {
      throw std::runtime_error("Dual pow() invalid for negative base and non-integer exponent.");
    }
    Dual out;
    out.value_ = std::pow(a.value_, p);
    out.grad_.resize(a.grad_.size(), 0.0);
    const double factor = p * std::pow(a.value_, p - 1.0);
    for (size_t i = 0; i < a.grad_.size(); ++i) {
      out.grad_[i] = factor * a.grad_[i];
    }
    return out;
  }

 private:
  double value_ = 0.0;
  std::vector<double> grad_;
};

// -------- Reverse-mode AutoDiff (tape as computation graph) --------
struct RevNode {
  double value = 0.0;
  double grad = 0.0;
  std::vector<std::pair<std::shared_ptr<RevNode>, double>> parents;
};

class RevVar {
 public:
  RevVar() : node_(std::make_shared<RevNode>()) {}
  explicit RevVar(double value) : node_(std::make_shared<RevNode>()) { node_->value = value; }
  explicit RevVar(std::shared_ptr<RevNode> node) : node_(std::move(node)) {}

  static RevVar variable(double value) { return RevVar(value); }

  double value() const { return node_->value; }
  std::shared_ptr<RevNode> node() const { return node_; }

 private:
  std::shared_ptr<RevNode> node_;
};

std::shared_ptr<RevNode> make_node(
    double value, const std::vector<std::pair<std::shared_ptr<RevNode>, double>>& parents) {
  auto node = std::make_shared<RevNode>();
  node->value = value;
  node->parents = parents;
  return node;
}

RevVar operator+(const RevVar& a, const RevVar& b) {
  return RevVar(make_node(a.value() + b.value(), {{a.node(), 1.0}, {b.node(), 1.0}}));
}

RevVar operator-(const RevVar& a, const RevVar& b) {
  return RevVar(make_node(a.value() - b.value(), {{a.node(), 1.0}, {b.node(), -1.0}}));
}

RevVar operator*(const RevVar& a, const RevVar& b) {
  return RevVar(make_node(a.value() * b.value(), {{a.node(), b.value()}, {b.node(), a.value()}}));
}

RevVar operator/(const RevVar& a, const RevVar& b) {
  if (std::abs(b.value()) < 1e-15) {
    throw std::runtime_error("RevVar division by near-zero.");
  }
  const double b2_inv = 1.0 / (b.value() * b.value());
  return RevVar(make_node(a.value() / b.value(), {{a.node(), 1.0 / b.value()}, {b.node(), -a.value() * b2_inv}}));
}

RevVar operator-(const RevVar& a) { return RevVar(make_node(-a.value(), {{a.node(), -1.0}})); }

RevVar sin(const RevVar& a) { return RevVar(make_node(std::sin(a.value()), {{a.node(), std::cos(a.value())}})); }

RevVar cos(const RevVar& a) { return RevVar(make_node(std::cos(a.value()), {{a.node(), -std::sin(a.value())}})); }

RevVar exp(const RevVar& a) {
  const double v = std::exp(a.value());
  return RevVar(make_node(v, {{a.node(), v}}));
}

RevVar log(const RevVar& a) {
  if (a.value() <= 0.0) {
    throw std::runtime_error("RevVar log() domain error.");
  }
  return RevVar(make_node(std::log(a.value()), {{a.node(), 1.0 / a.value()}}));
}

RevVar pow(const RevVar& a, double p) {
  if (a.value() < 0.0 && std::abs(std::round(p) - p) > 1e-12) {
    throw std::runtime_error("RevVar pow() invalid for negative base and non-integer exponent.");
  }
  const double v = std::pow(a.value(), p);
  const double dv = p * std::pow(a.value(), p - 1.0);
  return RevVar(make_node(v, {{a.node(), dv}}));
}

void topo_dfs(const std::shared_ptr<RevNode>& node, std::unordered_set<const RevNode*>& visited,
              std::vector<std::shared_ptr<RevNode>>& topo) {
  if (!node) {
    return;
  }
  if (!visited.insert(node.get()).second) {
    return;
  }
  for (const auto& parent_info : node->parents) {
    topo_dfs(parent_info.first, visited, topo);
  }
  topo.push_back(node);
}

void backward(const RevVar& out) {
  std::unordered_set<const RevNode*> visited;
  std::vector<std::shared_ptr<RevNode>> topo;
  topo_dfs(out.node(), visited, topo);

  for (auto& node : topo) {
    node->grad = 0.0;
  }
  out.node()->grad = 1.0;

  for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
    const auto& node = *it;
    for (const auto& parent_info : node->parents) {
      parent_info.first->grad += parent_info.second * node->grad;
    }
  }
}

template <typename Func>
EvalResult evaluate_forward(const Func& f, const std::vector<double>& x) {
  std::vector<Dual> vars;
  vars.reserve(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    vars.emplace_back(x[i], x.size(), i);
  }
  const Dual out = f(vars);
  return EvalResult{out.value(), out.grad()};
}

template <typename Func>
EvalResult evaluate_reverse(const Func& f, const std::vector<double>& x) {
  std::vector<RevVar> vars;
  vars.reserve(x.size());
  for (double xi : x) {
    vars.push_back(RevVar::variable(xi));
  }
  const RevVar out = f(vars);
  backward(out);

  std::vector<double> grad(x.size(), 0.0);
  for (size_t i = 0; i < x.size(); ++i) {
    grad[i] = vars[i].node()->grad;
  }
  return EvalResult{out.value(), grad};
}

struct LineSearchOptions {
  double c1 = 1e-4;
  double c2 = 0.9;
  double initial_step = 1.0;
  double max_step = 20.0;
  int max_iterations = 30;
  int max_zoom_iterations = 30;
};

struct LineSearchResult {
  double alpha = 0.0;
  EvalResult eval;
  bool success = false;
  int iterations = 0;
};

using Evaluator = std::function<EvalResult(const std::vector<double>&)>;

LineSearchResult zoom(const Evaluator& eval, const std::vector<double>& x, const std::vector<double>& p, double phi0,
                      double derphi0, double alo, double ahi, EvalResult flo, const LineSearchOptions& options) {
  LineSearchResult result;
  result.alpha = alo;
  result.eval = flo;

  for (int i = 0; i < options.max_zoom_iterations; ++i) {
    const double aj = 0.5 * (alo + ahi);
    EvalResult fj = eval(add_scaled(x, p, aj));

    if (fj.value > phi0 + options.c1 * aj * derphi0 || fj.value >= flo.value) {
      ahi = aj;
    } else {
      const double derphij = dot(fj.grad, p);
      if (std::abs(derphij) <= -options.c2 * derphi0) {
        return LineSearchResult{aj, fj, true, i + 1};
      }
      if (derphij * (ahi - alo) >= 0.0) {
        ahi = alo;
      }
      alo = aj;
      flo = fj;
    }

    result.alpha = aj;
    result.eval = fj;
    result.iterations = i + 1;
    if (std::abs(ahi - alo) < 1e-14) {
      break;
    }
  }
  return result;
}

LineSearchResult strong_wolfe_line_search(const Evaluator& eval, const std::vector<double>& x,
                                          const std::vector<double>& p, const EvalResult& fx,
                                          const LineSearchOptions& options) {
  const double phi0 = fx.value;
  const double derphi0 = dot(fx.grad, p);
  if (derphi0 >= 0.0) {
    return LineSearchResult{0.0, fx, false, 0};
  }

  double a_prev = 0.0;
  EvalResult f_prev = fx;
  double a = options.initial_step;
  LineSearchResult latest{0.0, fx, false, 0};

  for (int i = 0; i < options.max_iterations; ++i) {
    a = std::min(a, options.max_step);
    EvalResult f_a = eval(add_scaled(x, p, a));
    latest = LineSearchResult{a, f_a, false, i + 1};

    if (f_a.value > phi0 + options.c1 * a * derphi0 || (i > 0 && f_a.value >= f_prev.value)) {
      auto z = zoom(eval, x, p, phi0, derphi0, a_prev, a, f_prev, options);
      z.iterations += i + 1;
      return z;
    }

    const double derphi = dot(f_a.grad, p);
    if (std::abs(derphi) <= -options.c2 * derphi0) {
      return LineSearchResult{a, f_a, true, i + 1};
    }

    if (derphi >= 0.0) {
      auto z = zoom(eval, x, p, phi0, derphi0, a_prev, a, f_prev, options);
      z.iterations += i + 1;
      return z;
    }

    a_prev = a;
    f_prev = f_a;
    a = std::min(2.0 * a, options.max_step);
  }
  return latest;
}

LineSearchResult armijo_backtracking(const Evaluator& eval, const std::vector<double>& x,
                                     const std::vector<double>& p, const EvalResult& fx,
                                     double initial_alpha = 1.0, double c = 1e-4, double tau = 0.5,
                                     int max_iterations = 25) {
  LineSearchResult latest{0.0, fx, false, 0};
  const double derphi0 = dot(fx.grad, p);
  if (derphi0 >= 0.0) {
    return latest;
  }

  double alpha = initial_alpha;
  for (int i = 0; i < max_iterations; ++i) {
    EvalResult f_trial = eval(add_scaled(x, p, alpha));
    latest = LineSearchResult{alpha, f_trial, false, i + 1};
    if (f_trial.value <= fx.value + c * alpha * derphi0) {
      latest.success = true;
      return latest;
    }
    alpha *= tau;
  }
  return latest;
}

struct OptimizationOptions {
  int max_iterations = 300;
  double grad_tolerance = 1e-6;
  LineSearchOptions line_search;
  int lbfgs_memory = 8;
};

struct OptimizationResult {
  std::vector<double> x;
  double value = 0.0;
  std::vector<double> grad;
  int iterations = 0;
  bool converged = false;
};

using Matrix = std::vector<std::vector<double>>;

Matrix identity_matrix(size_t n) {
  Matrix I(n, std::vector<double>(n, 0.0));
  for (size_t i = 0; i < n; ++i) {
    I[i][i] = 1.0;
  }
  return I;
}

std::vector<double> mat_vec(const Matrix& A, const std::vector<double>& x) {
  if (A.size() != x.size()) {
    throw std::invalid_argument("mat_vec(): matrix and vector sizes do not match.");
  }
  std::vector<double> out(x.size(), 0.0);
  for (size_t i = 0; i < A.size(); ++i) {
    for (size_t j = 0; j < x.size(); ++j) {
      out[i] += A[i][j] * x[j];
    }
  }
  return out;
}

template <typename Func>
OptimizationResult gradient_descent(const Func& f, const std::vector<double>& x0, const OptimizationOptions& options) {
  const Evaluator eval = [&f](const std::vector<double>& x) { return evaluate_reverse(f, x); };

  std::vector<double> x = x0;
  EvalResult fx = eval(x);
  bool converged = false;
  int it = 0;

  for (; it < options.max_iterations; ++it) {
    if (norm(fx.grad) < options.grad_tolerance) {
      converged = true;
      break;
    }
    std::vector<double> p = scale(fx.grad, -1.0);
    LineSearchResult ls = strong_wolfe_line_search(eval, x, p, fx, options.line_search);
    if (!ls.success) {
      ls = armijo_backtracking(eval, x, p, fx, options.line_search.initial_step);
    }
    const double alpha = ls.success ? ls.alpha : 1e-3;
    x = add_scaled(x, p, alpha);
    fx = eval(x);
  }

  return OptimizationResult{x, fx.value, fx.grad, it, converged};
}

template <typename Func>
OptimizationResult bfgs_optimize(const Func& f, const std::vector<double>& x0, const OptimizationOptions& options) {
  const Evaluator eval = [&f](const std::vector<double>& x) { return evaluate_reverse(f, x); };

  std::vector<double> x = x0;
  EvalResult fx = eval(x);
  const size_t n = x.size();
  Matrix H = identity_matrix(n);
  bool converged = false;
  int it = 0;

  for (; it < options.max_iterations; ++it) {
    if (norm(fx.grad) < options.grad_tolerance) {
      converged = true;
      break;
    }

    std::vector<double> p = scale(mat_vec(H, fx.grad), -1.0);
    if (dot(p, fx.grad) >= 0.0) {
      H = identity_matrix(n);
      p = scale(fx.grad, -1.0);
    }

    LineSearchResult ls = strong_wolfe_line_search(eval, x, p, fx, options.line_search);
    if (!ls.success) {
      ls = armijo_backtracking(eval, x, p, fx, options.line_search.initial_step);
    }
    const double alpha = ls.success ? ls.alpha : 1e-3;

    std::vector<double> x_next = add_scaled(x, p, alpha);
    EvalResult fx_next = eval(x_next);

    const std::vector<double> s = subtract(x_next, x);
    const std::vector<double> y = subtract(fx_next.grad, fx.grad);
    const double ys = dot(y, s);

    if (ys > 1e-12) {
      const std::vector<double> Hy = mat_vec(H, y);
      const double yHy = dot(y, Hy);
      const double factor = (1.0 + yHy / ys) / ys;
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
          H[i][j] += factor * s[i] * s[j] - (Hy[i] * s[j] + s[i] * Hy[j]) / ys;
        }
      }
    } else {
      H = identity_matrix(n);
    }

    x = x_next;
    fx = fx_next;
  }

  return OptimizationResult{x, fx.value, fx.grad, it, converged};
}

std::vector<double> lbfgs_two_loop(const std::vector<double>& g, const std::vector<std::vector<double>>& s_hist,
                                   const std::vector<std::vector<double>>& y_hist) {
  const size_t k = s_hist.size();
  std::vector<double> q = g;
  std::vector<double> alpha(k, 0.0);
  std::vector<double> rho(k, 0.0);

  for (size_t i = 0; i < k; ++i) {
    rho[i] = 1.0 / dot(y_hist[i], s_hist[i]);
  }

  for (int i = static_cast<int>(k) - 1; i >= 0; --i) {
    alpha[i] = rho[i] * dot(s_hist[i], q);
    q = subtract(q, scale(y_hist[i], alpha[i]));
  }

  double gamma = 1.0;
  if (k > 0) {
    const auto& s_last = s_hist.back();
    const auto& y_last = y_hist.back();
    gamma = dot(s_last, y_last) / dot(y_last, y_last);
  }
  std::vector<double> r = scale(q, gamma);

  for (size_t i = 0; i < k; ++i) {
    const double beta = rho[i] * dot(y_hist[i], r);
    r = add(r, scale(s_hist[i], alpha[i] - beta));
  }
  return scale(r, -1.0);
}

template <typename Func>
OptimizationResult lbfgs_optimize(const Func& f, const std::vector<double>& x0, const OptimizationOptions& options) {
  const Evaluator eval = [&f](const std::vector<double>& x) { return evaluate_reverse(f, x); };

  std::vector<double> x = x0;
  EvalResult fx = eval(x);
  std::vector<std::vector<double>> s_hist;
  std::vector<std::vector<double>> y_hist;
  bool converged = false;
  int it = 0;

  for (; it < options.max_iterations; ++it) {
    if (norm(fx.grad) < options.grad_tolerance) {
      converged = true;
      break;
    }

    std::vector<double> p = s_hist.empty() ? scale(fx.grad, -1.0) : lbfgs_two_loop(fx.grad, s_hist, y_hist);
    if (dot(p, fx.grad) >= 0.0) {
      p = scale(fx.grad, -1.0);
      s_hist.clear();
      y_hist.clear();
    }

    LineSearchResult ls = armijo_backtracking(eval, x, p, fx, options.line_search.initial_step);
    if (!ls.success) {
      ls = strong_wolfe_line_search(eval, x, p, fx, options.line_search);
    }
    const double alpha = ls.success ? ls.alpha : 1e-3;

    std::vector<double> x_next = add_scaled(x, p, alpha);
    EvalResult fx_next = eval(x_next);

    std::vector<double> s = subtract(x_next, x);
    std::vector<double> y = subtract(fx_next.grad, fx.grad);
    const double ys = dot(y, s);
    if (ls.success && ys > 1e-12) {
      if (static_cast<int>(s_hist.size()) == options.lbfgs_memory) {
        s_hist.erase(s_hist.begin());
        y_hist.erase(y_hist.begin());
      }
      s_hist.push_back(std::move(s));
      y_hist.push_back(std::move(y));
    } else if (!ls.success || ys <= 1e-12) {
      s_hist.clear();
      y_hist.clear();
    }

    x = x_next;
    fx = fx_next;
  }

  return OptimizationResult{x, fx.value, fx.grad, it, converged};
}

struct Rosenbrock {
  template <typename T>
  T operator()(const std::vector<T>& x) const {
    if (x.size() < 2) {
      throw std::invalid_argument("Rosenbrock requires at least 2 dimensions.");
    }

    T total = T(0.0);
    for (size_t i = 0; i + 1 < x.size(); ++i) {
      const T a = x[i + 1] - x[i] * x[i];
      const T b = T(1.0) - x[i];
      total = total + T(100.0) * a * a + b * b;
    }
    return total;
  }
};

void print_result(const char* method, const OptimizationResult& r) {
  std::cout << method << "\n";
  std::cout << "  converged: " << (r.converged ? "true" : "false") << "\n";
  std::cout << "  iterations: " << r.iterations << "\n";
  std::cout << "  f(x): " << r.value << "\n";
  std::cout << "  ||grad||: " << norm(r.grad) << "\n";
  std::cout << "  x*: ";
  print_vector(r.x);
  std::cout << "\n\n";
}

}  // namespace opt

int main() {
  using namespace opt;

  std::cout << std::fixed << std::setprecision(6);

  Rosenbrock rosenbrock;
  const std::vector<double> x0 = {-1.2, 1.0};

  const EvalResult fwd = evaluate_forward(rosenbrock, x0);
  const EvalResult rev = evaluate_reverse(rosenbrock, x0);

  std::cout << "=== AutoDiff check on Rosenbrock(x0) ===\n";
  std::cout << "x0 = ";
  print_vector(x0);
  std::cout << "\n";
  std::cout << "Forward mode value: " << fwd.value << "\n";
  std::cout << "Forward mode grad : ";
  print_vector(fwd.grad);
  std::cout << "\n";
  std::cout << "Reverse mode value: " << rev.value << "\n";
  std::cout << "Reverse mode grad : ";
  print_vector(rev.grad);
  std::cout << "\n\n";

  OptimizationOptions options;
  options.max_iterations = 400;
  options.grad_tolerance = 1e-8;
  options.line_search.c1 = 1e-4;
  options.line_search.c2 = 0.9;
  options.line_search.initial_step = 1.0;
  options.line_search.max_step = 20.0;
  options.lbfgs_memory = 6;

  const OptimizationResult gd = gradient_descent(rosenbrock, x0, options);
  const OptimizationResult bfgs = bfgs_optimize(rosenbrock, x0, options);
  const OptimizationResult lbfgs = lbfgs_optimize(rosenbrock, x0, options);

  std::cout << "=== Optimization results ===\n";
  print_result("Gradient Descent + Wolfe line search", gd);
  print_result("BFGS + Wolfe line search", bfgs);
  print_result("L-BFGS + Wolfe line search", lbfgs);

  return 0;
}
