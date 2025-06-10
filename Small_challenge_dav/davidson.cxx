#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

VectorXd lambda;
MatrixXd V;
double L = 10.0;

size_t matrix_vector_products_count = 0;

void davidson(MatrixXd H, VectorXd v1, size_t M, size_t iterations_count = 20) {
  size_t N = H.rows();
  // enter start guess as first columns
  V = MatrixXd(N,1);
  V.col(0) = v1;
  lambda = VectorXd(1);
  // and its Rayleigh quotient as the guess for its eigenvalue
  lambda[0] = v1.dot(H * v1);
  // file for plotting
  ofstream psi_file("psi.dat");
  for (size_t iteration = 0; iteration < iterations_count; ++iteration) {
    size_t n = V.cols();
    MatrixXd W(N,2*n);
    for (size_t i = 0; i < n; ++i) {
      W.col(i) = V.col(i);
      ++matrix_vector_products_count;
      VectorXd residuum = H * V.col(i) - lambda[i] * V.col(i);
      VectorXd delta_v(N);
      for (size_t k = 0; k < N; ++k) {
        delta_v[k] = -residuum[k] / (H(k,k) - lambda[i]);
      }
      W.col(n+i) = delta_v;
    }
    // orthogonalize old and new trial vectors
    HouseholderQR<MatrixXd> W_QR(W);
    // matrix Q contains orthogonal columns with identical span
    W = W_QR.householderQ() * MatrixXd::Identity(N,2*n);

    // project H onto subspace
    MatrixXd h(2*n,2*n);
    for (size_t j = 0; j < 2*n; ++j) {
      ++matrix_vector_products_count;
      VectorXd Hb = H * W.col(j);
      // loop over distinct values (i,j)
      for (size_t i = 0; i <= j; ++i) {
        h(i,j) = W.col(i).dot(Hb);
        if (i < j) {
          h(j,i) = h(i,j);
        }
      }
    }
    // diagonalize in subspace
    SelfAdjointEigenSolver<MatrixXd> h_eigensolver(h);
    // take first M columns, corresponding to lowest eigenvalues
    size_t m = min(M,2*n);
    lambda = MatrixXd::Identity(m,2*n) *  h_eigensolver.eigenvalues();
    V = W * h_eigensolver.eigenvectors() * MatrixXd::Identity(2*n,m);

    for (size_t k = 0; k < N; ++k) {
      // plot ground state and first excited state
      psi_file << k*L/N << " " << V(k,0) << " " << V(k,1) << endl;
    }
    psi_file << endl << endl;
  }

}

int main() {
  size_t N = 100;
  MatrixXd H = MatrixXd::Zero(N,N);
  double DeltaX_squared = pow(L/N,2);
  for (size_t k = 0; k < N; ++k) {
    // three point stencil for laplacian
    H(k,k) = +1.0 / DeltaX_squared;
    H(k,(k+1)%N) = -0.5 / DeltaX_squared;
    H(k,(k+N-1)%N) = -0.5 / DeltaX_squared;
    // anharmonic potential
    double x = k*L/N - L/2;
    H(k,k) += pow(x,4)/24;
  }

  VectorXd v1(N);
  // trial guess:
  for (size_t k = 0; k < N; ++k) {
    // break symmetries to find odd and even eigenfunctions
    v1[k] = 1 + k*L/N;
  }
//  v1 = VectorXd::Random(N);
  v1.normalize();

  // Davidson with a 16-32 dimensional trial space and 16 iterations
  davidson(H, v1, 16, 8);

  SelfAdjointEigenSolver<MatrixXd> H_solver(H);
  VectorXd eigenvalues = H_solver.eigenvalues();

  cout << " Ritz values   Eigenvalues" << endl << scientific;
  for (size_t i = 0; i < lambda.size(); ++i) {
    // compare with numerically exact solution
    cout << lambda[i] << " " << eigenvalues[i] << endl;
  }
  cout << "matrix-vector products: " << matrix_vector_products_count << endl;
  cout << "psi.dat contains plots" << endl;
  return 0;
}
