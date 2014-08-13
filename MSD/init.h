#ifndef __init_h_
#define __init_h_

#include <string.h>
#include <vector>
using namespace std;

class Mol {
public:
  double lbox;
  vector<int> iatom;
  vector<std::string> ta;
  vector<std::string> tr;
  vector<std::string> tc;
  vector<int> ires;
  vector< vector<double> > r;
  vector< vector<int> > igrid; 
};

class Hbond {
public:
  vector<int> iatom; // atom number of the target atom
  vector<int> igroup; // 1 for phosphate, 2 for ribose, 3 for base
  vector<int> itype; // 0 for acceptor, 1 for donor
  vector< vector<int> > hatom; // atom number of hydrogen of water that h-bond
  vector< vector<int> > hbfreq; // frequency of h-bond with hatom 
  vector< vector<int> > hstart; // the frame number when starting h-bond with hatom
  vector< vector<int> > hend; // the frame number when h-bond with hatom ends
  vector< vector<int> > hdiff;
  vector< vector<int> > uatom; // atom number of water that is within the certain range where the hydrogen bond with the target atom might forms, but does not form. In other words, the distance criteria is satisfied, but the angle criteria is not satisfied.
  vector< vector<int> > ubfreq; 
  vector< vector<int> > ustart;
  vector< vector<int> > uend;
  vector< vector<int> > ua; // uatom for at the current snapshot, will clear before reading the next snapshot information
};

class Counter {
public:
  vector<int> iatom; // atom number of the target ion
  vector<int> patom; // atom number of the closest phospher atom
  vector< vector<double> > d_ip; // distance between the ion and the closest P atom
  vector< vector<int> > oatom; // atom number of oxygen of water that directly bind to the target ion for all the snapshots;
  vector< vector<int> > obfreq; // frequency of directly bond with oatom
  vector< vector<int> > ostart; // the frame number when starting directly binding to oatom
  vector< vector<int> > oend; // the frame number when directly bond with oatom ends
  vector< vector<int> > oatemp; // atom number of oxygen of water that directly bind to the target ion at the current snapshot
  vector<int> ifrna; // check if the bound water of ion also H-bonds directly to RNA, 1 for yes, 0 for no
  vector< vector<int> > ratom; // atom number of rna that directly bound to target ion
  vector< vector<int> > rbfreq;
  vector< vector<int> > rstart;
  vector< vector<int> > rend;
};

void read_pdb(string,Mol&,Mol&,Mol&,Mol&,Mol&, vector< vector<int> >&,vector< vector<int> >&);
void cal_watlist(Mol&, Mol&, int&, vector<int>&);
void cal_RW_scatter(Mol&,vector< vector<int> >&,vector< vector<int> >&,int&,vector<int>&,double&);
double dist2(vector<double>,vector<double>);
double cosine(vector<double>,vector<double>,vector<double>);
bool fexists(string);
#endif

