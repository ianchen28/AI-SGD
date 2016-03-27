// -*- C++ -*-
// SVM with stochastic gradient
// Copyright (C) 2007- Leon Bottou
//
// SVM using SGD with implicit Updates.
// Copyright (C) 2014- Panos Toulis (ptoulis@fas.harvard.edu)
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111, USA
//
// ADDENDUM: This is an implementation of the implicit update method for SVM model.
// The main part of this code was written by Leon Bottou.
// The modifications for the implicit updates are in function trainOne (mostly)
// and are due to Panos Toulis. This is based on (Toulis et.al., ICML 2014)
// Email ptoulis@fas.harvard.edu if you have questions.
//
//
// Example: ./svmimplicit -lambda 5e-7 -epochs 12 rcv1.train.bin.gz
//  to active loss compile with -DLOSS=<lossfunctionclass> (see README for more).

#include "assert.h"
#include "data.h"
#include "gzstream.h"
#include "loss.h"
#include "timer.h"
#include "vectors.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <map>
#include <string>
#include <vector>

using namespace std;

// ---- Loss function

// Compile with -DLOSS=xxxx to define the loss function.
// Loss functions are defined in file loss.h)
#ifndef LOSS
# define LOSS LogLoss
#endif

// ---- Bias term

// Compile with -DBIAS=[1/0] to enable/disable the bias term.
// Compile with -DREGULARIZED_BIAS=1 to enable regularization on the bias term

#ifndef BIAS
# define BIAS 1
#endif
#ifndef REGULARIZED_BIAS
# define REGULARIZED_BIAS 0
#endif

// ---- Plain stochastic gradient descent with implicit updates.
class SvmIsgd
{
    public:
        SvmIsgd(int dim, double lambda, double eta0  = 0);
        void renorm();
        double wnorm();
        double testOne(const SVector &x, double y, double *ploss, double *pnerr);
        void trainOne(const SVector &x, double y, double eta);
    public:
        void train(int imin, int imax, const xvec_t &x, const yvec_t &y, const char *prefix = "");
        void test(int imin, int imax, const xvec_t &x, const yvec_t &y, const char *prefix = "");
    public:
        double evaluateEta(int imin, int imax, const xvec_t &x, const yvec_t &y, double eta);
        void determineEta0(int imin, int imax, const xvec_t &x, const yvec_t &y);
    private:
        double  lambda;
        double  eta0;
        FVector w;
        double  wDivisor;
        double  wBias;
        double  t;
};

/// Constructor
SvmIsgd::SvmIsgd(int dim, double lambda, double eta0)
: lambda(lambda), eta0(eta0),
    w(dim), wDivisor(1), wBias(0),
    t(0)
{
}

/// Renormalize the weights
void SvmIsgd::renorm()
{
    cout << "\n\t Renormalizing...";
    if (wDivisor != 1.0)
    {
        w.scale(1.0 / wDivisor);
        wDivisor = 1.0;
    }
}

/// Compute the norm of the weights
double SvmIsgd::wnorm()
{
    double norm = dot(w,w) / wDivisor / wDivisor;
#if REGULARIZED_BIAS
    norm += wBias * wBias
#endif
        return norm;
}

/// Compute the output for one example.
double SvmIsgd::testOne(const SVector &x, double y, double *ploss, double *pnerr)
{
    double s = dot(w,x) / wDivisor + wBias;
    // cout << s << " real " << y << "\n";
    if (ploss)
        *ploss += LOSS::loss(s, y);
    if (pnerr)
        *pnerr += (s * y <= 0) ? 1 : 0;
    return s;
}

/// Perform one iteration of the Implicit algorithm with specified gains
/// This is the only function differentiating the implicit with the classical
/// SGD implementation.
void SvmIsgd::trainOne(const SVector &x, double y, double eta)
{
    // HingeLoss case.
    if(LOSS::name().compare("HingeLoss")==0) 
    {
        double ypred = dot(x, w) / wDivisor;
        double implicitFactor = (1 + lambda * eta);
        if(1 - y * ypred / implicitFactor < 0)
        {
            wDivisor *= implicitFactor;
            // Update will be W_n+1 = Wn  / (1+lambda * eta)
        }
        else
        {
            double ypred = 0;  // computes x_t' theta_{t+1} (next update)
            for(const SVector::Pair *p = x; p->i >= 0; p++)
            {
                double w_i = w.get(p->i) / wDivisor;
                ypred += p->v * (w_i + p->v * eta * y);
            }

            if(1 - y * ypred / implicitFactor >= 0)
            {
                w.add(x, eta * y * wDivisor);
                wDivisor *= implicitFactor;
                // Update should be theta_{t+!1} = (1/(1+lambda eta)) * (theta_t + eta * yt * xt)
            }
            else
            {
                // do nothing (no update in parameters).
            }
        }

        if (wDivisor > 1e5) renorm();
    }

    else if(LOSS::name().compare("LogLoss")==0) {
        // Need to solve  ξ_t = at (yt - h(theta_t' xt + ξt ||xt||^2))
        // Solve approximately by using
        // ξt = (1 / (1 + at ||xt||^2 h'(theta_t'xt)) * at * (yt - h(theta_t' xt))
        // TODO(ptoulis): Use implicit Algorithm 1 of (Toulis, et.al., ICML14)
        double wx = dot(w, x) / wDivisor;
        double ypred = 2 * (exp(wx) / (1 + exp(wx))) - 1;
        double implicitFactor = 1 + eta * dot(x, x) * ypred / (1 + exp(wx));

        double ksi_t = (1 / implicitFactor) * eta * (y - ypred);
        w.add(x,  wDivisor * ksi_t);
    }

    else {
        cout << "#" << LOSS::name() << "# -- loss not found.";
    }
}

/// Perform a training epoch
void SvmIsgd::train(int imin, int imax, const xvec_t &xp, const yvec_t &yp, const char *prefix)
{
    cout << prefix << "Training on [" << imin << ", " << imax << "]." << endl;
    assert(imin <= imax);
    assert(eta0 > 0);
    for (int i=imin; i<=imax; i++)
    {
        double eta = eta0 / (1 + lambda * eta0 * t);
        trainOne(xp.at(i), yp.at(i), eta);
        t += 1;
    }
    cout << prefix << setprecision(6) << "wNorm=" << wnorm();
#if BIAS
    cout << " wBias=" << wBias;
#endif
    cout << endl;
}

/// Perform a test pass
void SvmIsgd::test(int imin, int imax, const xvec_t &xp, const yvec_t &yp, const char *prefix)
{
    cout << prefix << "Testing on [" << imin << ", " << imax << "]." << endl;
    assert(imin <= imax);
    double nerr = 0;
    double loss = 0;
    for (int i=imin; i<=imax; i++)
        testOne(xp.at(i), yp.at(i), &loss, &nerr);
    nerr = nerr / (imax - imin + 1);
    loss = loss / (imax - imin + 1);
    double cost = loss + 0.5 * lambda * wnorm();
    cout << prefix
        << "Loss=" << setprecision(12) << loss
        << " Cost=" << setprecision(12) << cost
        << " Misclassification=" << setprecision(4) << 100 * nerr << "%."
        << endl;
}

/// Perform one epoch with fixed eta and return cost
double SvmIsgd::evaluateEta(int imin, int imax, const xvec_t &xp, const yvec_t &yp, double eta)
{
    SvmIsgd clone(*this); // take a copy of the current state
    assert(imin <= imax);
    for (int i=imin; i<=imax; i++)
        clone.trainOne(xp.at(i), yp.at(i), eta);
    double loss = 0;
    double cost = 0;
    for (int i=imin; i<=imax; i++)
        clone.testOne(xp.at(i), yp.at(i), &loss, 0);
    loss = loss / (imax - imin + 1);
    cost = loss + 0.5 * lambda * clone.wnorm();
    // cout << "Trying eta=" << eta << " yields cost " << cost << endl;
    return cost;
}

void SvmIsgd::determineEta0(int imin, int imax, const xvec_t &xp, const yvec_t &yp)
{
    const double factor = 2.0;
    double loEta = 1;
    double loCost = evaluateEta(imin, imax, xp, yp, loEta);
    double hiEta = loEta * factor;
    double hiCost = evaluateEta(imin, imax, xp, yp, hiEta);
    if (loCost < hiCost)
        while (loCost < hiCost)
        {
            hiEta = loEta;
            hiCost = loCost;
            loEta = hiEta / factor;
            loCost = evaluateEta(imin, imax, xp, yp, loEta);
        }
    else if (hiCost < loCost)
        while (hiCost < loCost)
        {
            loEta = hiEta;
            loCost = hiCost;
            hiEta = loEta * factor;
            hiCost = evaluateEta(imin, imax, xp, yp, hiEta);
        }
    eta0 = loEta;
    cout << "# Using eta0=" << eta0 << endl;
}

// --- Command line arguments
const char *trainfile = 0;
const char *testfile = 0;
bool normalize = true;
double lambda = 1e-5;
int epochs = 5;
int maxtrain = -1;

void usage(const char *progname)
{
    const char *s = ::strchr(progname,'/');
    progname = (s) ? s + 1 : progname;
    cerr << "Usage: " << progname << " [options] trainfile [testfile]" << endl
        << "Options:" << endl;
#define NAM(n) "    " << setw(16) << left << n << setw(0) << ": "
#define DEF(v) " (default: " << v << ".)"
    cerr << NAM("-lambda x")
        << "Regularization parameter" << DEF(lambda) << endl
        << NAM("-epochs n")
        << "Number of training epochs" << DEF(epochs) << endl
        << NAM("-dontnormalize")
        << "Do not normalize the L2 norm of patterns." << endl
        << NAM("-maxtrain n")
        << "Restrict training set to n examples." << endl;
#undef NAM
#undef DEF
    ::exit(10);
}

void parse(int argc, const char **argv)
{
    for (int i=1; i<argc; i++)
    {
        const char *arg = argv[i];
        if (arg[0] != '-')
        {
            if (trainfile == 0)
                trainfile = arg;
            else if (testfile == 0)
                testfile = arg;
            else
                usage(argv[0]);
        }
        else
        {
            while (arg[0] == '-')
                arg += 1;
            string opt = arg;
            if (opt == "lambda" && i+1<argc)
            {
                lambda = atof(argv[++i]);
                assert(lambda>0 && lambda<1e4);
            }
            else if (opt == "epochs" && i+1<argc)
            {
                epochs = atoi(argv[++i]);
                assert(epochs>0 && epochs<1e6);
            }
            else if (opt == "dontnormalize")
            {
                normalize = false;
            }
            else if (opt == "maxtrain" && i+1 < argc)
            {
                maxtrain = atoi(argv[++i]);
                assert(maxtrain > 0);
            }
            else
            {
                cerr << "Option " << argv[i] << " not recognized." << endl;
                usage(argv[0]);
            }

        }
    }
    if (! trainfile)
        usage(argv[0]);
}

void config(const char *progname)
{
    cout << "# Running: " << progname;
    cout << " -lambda " << lambda;
    cout << " -epochs " << epochs;
    if (! normalize) cout << " -dontnormalize";
    if (maxtrain > 0) cout << " -maxtrain " << maxtrain;
    cout << endl;
#define NAME(x) #x
#define NAME2(x) NAME(x)
    cout << "# Compiled with: "
        << " -DLOSS=" << NAME2(LOSS)
        << " -DBIAS=" << BIAS
        << " -DREGULARIZED_BIAS=" << REGULARIZED_BIAS
        << endl;
}

// --- main function
int dims;
xvec_t xtrain;
yvec_t ytrain;
xvec_t xtest;
yvec_t ytest;

int main(int argc, const char **argv)
{
    parse(argc, argv);
    config(argv[0]);
    if (trainfile)
        load_datafile(trainfile, xtrain, ytrain, dims, normalize, maxtrain);
    if (testfile)
        load_datafile(testfile, xtest, ytest, dims, normalize);
    cout << "# Number of features " << dims << "." << endl;
    // prepare svm
    int imin = 0;
    int imax = xtrain.size() - 1;
    int tmin = 0;
    int tmax = xtest.size() - 1;
    SvmIsgd svm(dims, lambda);
    Timer timer;
    // determine eta0 using sample
    int smin = 0;
    int smax = imin + min(1000, imax);
    timer.start();
    svm.determineEta0(smin, smax, xtrain, ytrain);
    timer.stop();
    // train
    for(int i=0; i<epochs; i++)
    {
        cout << "--------- Epoch " << i+1 << "." << endl;
        timer.start();
        svm.train(imin, imax, xtrain, ytrain);
        timer.stop();
        cout << "Total training time " << setprecision(6)
            << timer.elapsed() << " secs." << endl;
        svm.test(imin, imax, xtrain, ytrain, "train: ");
        if (tmax >= tmin)
            svm.test(tmin, tmax, xtest, ytest, "test:  ");
    }
    return 0;
}
