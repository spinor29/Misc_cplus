// Neural Network learning
// This file includes code for calculating the cost function of a neural
// network with one hidden layer.
//
// The execution needs two input files with training data and test data.
// To compile the code, it needs the Eigen library for the matrix calculation.

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <time.h>
#include <random>
#include <math.h>
#include <Eigen/Dense>
#include <nlopt.h>
//#include <dlib/optimization.h>

typedef std::vector<double> record_t;
typedef std::vector<record_t> data_t;

// Overload the stream input operator to read a list of CSV fields in a CSV record
std::istream& operator >> (std::istream& ins, record_t& record) {
    record.clear();
    
    // read the entire line into a string
    std::string line;
    getline(ins, line);
    
    // Use a 'stringstream' to separate the fields out of the line
    std::stringstream ss(line);
    std::string field;
    while (getline(ss, field, ',')) {
        std::stringstream fs(field);
        double f = 0.0;
        fs >> f;
        record.push_back(f);
    }
    
    return ins;
}

// Overload the stream input operator to read a list of CSV records in a CSV file
std::istream& operator >> (std::istream& ins, data_t& data) {
    data.clear();
    
    // Skip the header
    std::string header;
    getline(ins, header);
    
    // For every record read from the file, append it to the resulting 'data'
    record_t record;
    while (ins >> record) {
        data.push_back(record);
    }
    
    return ins;
}

void parse(std::string filename, data_t* data) {
    std::ifstream infile(filename);
    
    infile >> *data;
    
    if (!infile.eof()) {
        std::cout << "Something wrong with the file!" << std::endl;
        //return 1;
    }
    
    infile.close();
    
    return;
}

void randInitializeWeights(Eigen::VectorXd& nn_params, const int n_in, const int n_hid, const int n_lab) {
    const double epsilon_init = 0.12;
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0,1.0);
    
    for (int i = 0; i < nn_params.size(); i++) {
        double number = distribution(generator);
        nn_params(i) = number * 2.0 * epsilon_init - epsilon_init;
    }
    return;
}

void multi_a_bT(std::vector< std::vector<double> >& a,
         std::vector< std::vector<double> >& b, std::vector< std::vector<double> >& c) {
    // matrix multiplication between a and b.T
    // given matrix a and b, calculate c = a * b.T, where b.T is the transpose of b
    const int na1 = a.size();
    const int na2 = a[0].size();
    const int nb2 = b.size();
    const int nb1 = b[0].size();
    
    // initialize c with 0s
    // vector< vector<double> > c(na1, std::vector<double> (nb2, 0));
    
    for (int i = 0; i < na1; i++) {
        for (int j = 0; j < nb2; j++) {
            for (int k = 0; k < na2; k++) {
                c[i][j] += a[i][k] * b[j][k];
            }
        }
    }
    return;
}

void sigmoid(Eigen::MatrixXd& x, Eigen::MatrixXd& sig) {
    for (int i = 0; i < x.rows(); i++) {
        for (int j = 0; j < x.cols(); j++) {
            sig(i,j) = 1.0 / (1.0 + exp(x(i,j)));
        }
    }
}

//void sigmoidGradient(Eigen::MatrixXd& x, Eigen::MatrixXd& g) {
//    g = sigmoid(x).cwiseProduct((1.0-sigmoid(x)));
//}


void nnCostFunction(Eigen::VectorXd& nn_params, Eigen::MatrixXd& X0, Eigen::MatrixXd& y, const int n_input, const int n_hidden, const int n_labels, const double lambda) {

//double nnCostFunction(const column_vector& v, const int J_or_G) {
    
    
    Eigen::MatrixXd Theta1(n_hidden, n_input+1);
    Eigen::MatrixXd Theta2(n_labels, n_hidden+1);
    
    // Reshape nn_params to Theta1 and Theta2
    int k = 0;
    for (int i = 0; i < n_hidden; ++i) {
        for (int j = 0; j < n_input+1; ++j) {
            Theta1(i,j) = nn_params(k);
            k++;
        }
    }
    
    for (int i = 0; i < n_labels; ++i) {
        for (int j = 0; j < n_hidden+1; ++j) {
            Theta2(i,j) = nn_params(k);
            k++;
        }
    }
    
    const int m = X0.rows();
    
    // Feedforward
    
    Eigen::MatrixXd X(m, n_input+1);
    
    for (int i = 0; i < m ; ++i) {
        X(i,0) = 1.0;
        for (int j = 0; j < n_input; ++j) {
            X(i,j+1) = X0(i,j);
        }
    }
    
    Eigen::MatrixXd a2(m, n_hidden);
    
    time_t start = time(0);
    a2 = X * Theta1.transpose();
    time_t end = time(0);
    std::cout << "Eigen multiplication elapsed time: " << difftime(end, start) << "s" << std::endl;
    
    std::cout << a2(0,0) << " " << a2(0,1) << std::endl;
    std::cout << a2(1,0) << " " << a2(1,1) << std::endl;
    
    start = time(0);
    Eigen::MatrixXd sig_a2(m, n_hidden);
    sigmoid(a2, sig_a2);
    end = time(0);
    std::cout << "Sigmoid elapsed time: " << difftime(end, start) << "s" << std::endl;

    std::cout << sig_a2(0,0) << " " << sig_a2(0,1) << std::endl;
    std::cout << sig_a2(1,0) << " " << sig_a2(1,1) << std::endl;
    
    Eigen::MatrixXd a2_1(m, n_hidden+1);
    
    // Add a column of 1 to a2
    for (int i = 0; i < m; ++i) {
        a2_1(i,0) = 1.0;
        for (int j = 0; j < n_hidden; ++j) {
            a2_1(i,j+1) = sig_a2(i,j);
        }
    }

    Eigen::MatrixXd h(m, n_labels);
    Eigen::MatrixXd sig_h(m, n_labels);
    start = time(0);
    h = a2_1 * Theta2.transpose();
    end = time(0);
    std::cout << "h elapsed time: " << difftime(end, start) << "s" << std::endl;

    sigmoid(h,sig_h);
    
    Eigen::MatrixXd y2(m,1);
    
    double J = 0;
    
    Eigen::MatrixXd h_T(n_labels, m);
    h_T = sig_h.transpose();
    
    start = time(0);
    for (int c = 0; c < n_labels; c++) {
        for (int i = 0; i < m; i++) {
            y2(i,0) = (int(y(i,0)) == c);
            J += 1.0 / m * (log(h_T(c,i)) * y2(i,0) + log(1.0 - h_T(c,i)) * (1.0 - y2(i,0)));
        }
    }
    end = time(0);
    std::cout << "J elapsed time: " << difftime(end, start) << "s" << std::endl;
    
    // Regularization term
    Eigen::MatrixXd th1sq(n_hidden, n_input+1);
    Eigen::MatrixXd th2sq(n_labels, n_hidden+1);
    
    th1sq = Theta1.cwiseProduct(Theta1);
    th2sq = Theta2.cwiseProduct(Theta2);
    
    th1sq.col(0).setZero();
    th2sq.col(0).setZero();
    
    J += lambda / (2.0 * m) * (th1sq.sum() + th2sq.sum());
    std::cout << th1sq(0,0) << " " << th1sq(0,1) << std::endl;
    
    std::cout << "J = " << J << std::endl;
    
    //if (J_or_G == 0) return J;
    
    // If J_or_G != 0, continue to calculate gradients, and return gradients in column_vector format
    
    Eigen::MatrixXd Theta1_grad(n_hidden, n_input+1);
    Eigen::MatrixXd Theta2_grad(n_labels, n_hidden+1);
    
    // Backpropagation
    Eigen::MatrixXd a1(m,n_input+1);
    Eigen::MatrixXd z2(m,n_hidden);
    
    start = time(0);
    z2 = a1 * Theta1.transpose();
    end = time(0);
    std::cout << "z2 elapsed time: " << difftime(end, start) << "s" << std::endl;
    
    Eigen::MatrixXd sig_z2(m,n_hidden);
    
    start = time(0);
    sigmoid(z2,sig_z2);
    end = time(0);
    std::cout << "Sigmoid sig_z2 elapsed time: " << difftime(end, start) << "s" << std::endl;
    
    // Add a column of 1 to z2
    for (int i = 0; i < m; ++i) {
        a2_1(i,0) = 1.0;
        for (int j = 0; j < n_hidden; ++j) {
            a2_1(i,j+1) = sig_z2(i,j);
        }
    }

    start = time(0);
    Eigen::MatrixXd z3(m,n_labels);
    z3 = a2_1 * Theta2.transpose();
    Eigen::MatrixXd a3(m,n_labels);
    end = time(0);
    std::cout << "z3 elapsed time: " << difftime(end, start) << "s" << std::endl;
    
    sigmoid(z3, a3);
    
    Eigen::MatrixXd d3(m,n_labels);
    
    for (int i = 0; i < m; i++) {
        for (int c = 0; c < n_labels; c++) {
            d3(i,c) = a3(i,c) - (int(y(i,0)) == c);
        }
    }
    
    Eigen::MatrixXd d2_1(m,n_hidden+1);
    
    start = time(0);
    d2_1 = d3 * Theta2;
    end = time(0);
    std::cout << "d2_1 elapsed time: " << difftime(end, start) << "s" << std::endl;
    
    Eigen::MatrixXd d2(m,n_hidden);
    
    // Remove column 0 from d2_1
    for (int i = 0; i < m; ++i) {
        d2(i,0) = 1.0;
        for (int j = 0; j < n_hidden; ++j) {
            d2(i,j) = d2_1(i,j+1);
        }
    }
    
    Eigen::MatrixXd sigG(m, n_hidden);
 
    sigG = sig_z2.cwiseProduct((1.0 - sig_z2.array()).matrix());
    d2 = d2.cwiseProduct(sigG);
    
    start = time(0);
    Theta2_grad = d3.transpose() * a2_1;
    Theta1_grad = d2.transpose() * a1;
    
    Theta1_grad /= m;
    Theta2_grad /= m;
    
    Eigen::MatrixXd reg1(n_hidden, n_input+1);
    Eigen::MatrixXd reg2(n_labels, n_hidden+1);
    
    Eigen::MatrixXd Theta1_grad_reg(n_hidden, n_input+1);
    Eigen::MatrixXd Theta2_grad_reg(n_labels, n_hidden+1);
    
    reg1 = Theta1 * (lambda / double(m));
    reg2 = Theta2 * (lambda / double(m));
    
    reg1.col(0).setZero();
    reg2.col(0).setZero();
    
    Theta1_grad += reg1;
    Theta2_grad += reg2;
    
    end = time(0);
    std::cout << "grad elapsed time: " << difftime(end, start) << "s" << std::endl;
    
    return;
}

//double J_func(const column_vector& v) {
//    J = nnCostFunction(v, 0);
//    return J;
//}

//double G_func(const column_vector& v) {
//    grad = nnCostFunction(v, 1);
//    return grad;
//}

void solve_it(data_t& train_data) {
    const int n_labels = 10;
    const int m = train_data.size();
    const int ncols = train_data[0].size();
    const int n_input = ncols - 1; // input layer size
    const double scale = 255.0; // scaling factor for X to avoid exponential overflow
    const int n_hidden = 200;
    
    // Randomly shuffle the training data
    std::random_shuffle(train_data.begin(), train_data.end()); // use built-in random generator
    
    // Divide training data into training part and cross validation part
    const double ratio = 0.7; // ratio of size of examples for training to that of cross validation
    const int m1 = int(m * ratio);
    
    Eigen::MatrixXd y_train(m1,1);
    Eigen::MatrixXd X_train(m1,n_input);
    Eigen::MatrixXd y_cross(m1,1);
    Eigen::MatrixXd X_cross(m-m1,n_input);
    
    for (int i = 0; i < m1; ++i) {
        y_train(i,0) = train_data[i][0];
        for (int j = 0; j < n_input; ++j) {
            X_train(i,j) = train_data[i][j+1] / scale;
        }
    }
    
    //std::cout << y_train[0] << " " << y_train[1] << std::endl;
    
    for (int i = 0; i < (m-m1); ++i) {
        y_cross(i,0) = train_data[i][0];
        for (int j = 0; j < n_input; ++j) {
            X_cross(i,j) = train_data[i+m1][j+1] / scale;
        }
    }

    // Initialize the weights of the neural networks
    const int num_params = n_hidden * (n_input+1) + n_labels * (n_hidden + 1);
    //std::vector<double> initial_nn_params(num_params);
    Eigen::VectorXd initial_nn_params(num_params);
    
    randInitializeWeights(initial_nn_params, n_input, n_hidden, n_labels);
    
    // Backpropagation
    const double lambda = 1.0;

    nnCostFunction(initial_nn_params, X_train, y_train, n_input, n_hidden, n_labels, lambda);
    
	
	
    return;
}

int main(int argc, const char* argv[]) {
    std::string train_file = argv[1];
    std::string test_file = argv[2];
    
    // Load input data
    std::cout << "Loading data... " << train_file << " "  << test_file << std::endl;
    
    data_t train_data;
    
    time_t start = time(0);
    parse(train_file, &train_data);
    
    std::cout << "data size: " << train_data.size() << std::endl;
    std::cout << "data[0] size: " << train_data[0].size() << std::endl;
    std::cout << train_data[0][0] << " " << train_data[0][1] << std::endl;
    std::cout << train_data[1][0] << " " << train_data[1][1] << std::endl;
    time_t end = time(0);
    std::cout << "Elapsed time: " << difftime(end, start) << "s" << std::endl;
    
    // Solve the problem
    std::cout << "Solving problem..." << std::endl;
    start = time(0);
    solve_it(train_data);
    end = time(0);
    std::cout << "Elapsed time: " << difftime(end, start) << "s" << std::endl;
    
    return 0;
}
