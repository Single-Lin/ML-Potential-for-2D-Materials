/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(agni,PairAGNI)

#else

#ifndef LMP_PAIR_AGNI_H
#define LMP_PAIR_AGNI_H

#include "pair.h"
//#include <vector>
//#include <dlib/svm.h>

using namespace std;
//using namespace dlib;

namespace LAMMPS_NS {

class PairAGNI : public Pair {
 public:
  PairAGNI(class LAMMPS *);
  virtual ~PairAGNI();
  virtual void compute(int, int);
  void settings(int, char **);
  virtual void coeff(int, char **);
  //virtual double init_one(int, int);
  //virtual void init_style();
  void init_style();
  void init_list(int, class NeighList *);
  
  void grab(FILE *, int, double *);  //YY:2018-05-03
  
  double init_one(int, int);


  //**************** 2018-05-03 by YY***********//
  int pack_forward_comm(int, int *, double *, int, int *);
  void unpack_forward_comm(int, int, double *);
  int pack_reverse_comm(int, int, double *);
  void unpack_reverse_comm(int, int *, double *);
  double memory_usage();
  //**************** 2018-05-03 by YY***********//




    struct Param {
    double cut,cutsq,cut2,cutsq2,cut3,cutsq3;           // add cut2 and cutsq2  cut3 and cutsq3 by YY 
    double *eta,**xU,*yU,*epsilon,*Rho,*epeta;  //add an epsilon by YY
    //double *alpha;
    //double sigma,lambda,b;
    int numeta,numtrain,numtest,ielement,numepsilon,numRho,numepe;  //add numepsilon by YY
    };

 protected:
  double cutmax;                // max cutoff for all elements
  //double cutmax2;                // max cutoff2 for all elements
  int nelements;                // # of unique atom type labels
  int N_flag;                   // flag for the normalization 
  double Emax,Emin;
  double Fmax[3],Fmin[3],Smax[6],Smin[6];
  double **rho_v;
  char **elements;              // names of unique elements
  int *elem2param;              // mapping from element pairs to parameters
  int *map;                     // mapping from atom types to elements
  int nparams;                  // # of stored parameter sets
  Param *params;                // parameter set for an I-J interaction
  
  double *pe_arr;
  int pe_arr_len;
  
  //virtual void allocate();
  void read_file(char *);
  virtual void setup_params();
  

  ///**********************YY-0208-05-03**********************///
  /// Helper data structure for potential routine.
        struct MEAM2Body {
                int tag;
                double r;
                double fcut;
                double fcut_dev;
                double del[3];
        };

        int nmax;                              // Size of temporary array.
        int maxNeighbors;                      // The last maximum number of neighbors a single atoms has.
        MEAM2Body* twoBodyInfo;                // Temporary array.

        //void read_file(const char* filename);
        //void read_param(const char* filename);
        void allocate();
        double fun_cutoff(double, double);
        double fun_cutoff(double, double, double);
        double fun_cutoff_dev(double,double,double);
        double LJ_fpair(double, int );
        double LJ_Epair(double, int );
        double Hx(double );
        void   Data_Fitting();
        void   costheta_d(double,const double *, double,const double *, double, double *, double *);

  // vector functions, inline for efficiency
  inline double vec3_dot(const double *x,const double *y) {
    return x[0]*y[0] + x[1]*y[1] + x[2]*y[2];
  }
  inline void vec3_add(const double *x, const double *y, double *z) {
    z[0] = x[0]+y[0];  z[1] = x[1]+y[1];  z[2] = x[2]+y[2];
  }
  inline void vec3_scale(const double k, const double *x, double *y) {
    y[0] = k*x[0];  y[1] = k*x[1];  y[2] = k*x[2];
  }
  inline void vec3_scaleadd(const double k,const double *x,const double *y, double *z) {
    z[0] = k*x[0]+y[0];  z[1] = k*x[1]+y[1];  z[2] = k*x[2]+y[2];
  }


  //void Data_Fitting();
   
  int N_type; //current
  int CN1; //current
  //int mycol;
  //mycol = 121;
  //int mycol=CN1*N_type; //current
  double cconst;
  double pppp;     


 
    //typedef matrix<double,121,1> Fvect_Type; //current
    //Fvect_Type my_w,vv,fvect_dev,YY_means,YY_devs;
    //std::vector<Fvect_Type> fvect_arr;
	
    //std::vector<double> target_arr;
    //std::vector<Fvect_Type> fvect_test;
	
    //std::vector<double> target_test;
    double my_w[121],vv[121],fvect_dev[121],YY_means[121],YY_devs[121];
	
	
	/*********** krr ************
    typedef linear_kernel<Fvect_Type> kernel_type;
    vector_normalizer<Fvect_Type> normalizer;
    //typedef radial_basis_kernel<Fvect_Type> kernel_type;
    krr_trainer<kernel_type> trainer;
    //typedef decision_function<kernel_type> dec_funct_type;
    //typedef normalized_function<dec_funct_type> funct_type;
    //funct_type final_pot_trainer;
    decision_function<kernel_type> final_pot_trainer;	
    kernel_type kern;
    double cconst;	
	/*********** rls *************/
//	rls mytrainer;
  
  
  //  these two are for multi-elements 
  int getOrder(int, int, int);
  int getO(int, int, int);
  double myCos(double,double,double,double,double,double);
  
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: Cannot open AGNI potential file %s

The specified AGNI potential file cannot be opened.  Check that the path
and name are correct.

E: Incorrect format in AGNI potential file

The potential file is not compatible with the AGNI pair style
implementation in this LAMMPS version.

*/
