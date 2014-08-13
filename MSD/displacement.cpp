// Mean square displacement calculations.
// Given configurations of a series of snapshots in pdb format for a MD run
// for the RNA, calculate the mean square displacements, <x(t)-x(0)>, of
// the hydrogen atoms in RNA for all snapshots

#include<fstream>
#include<iostream>
#include<sstream>
#include<string>
#include<vector>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <ctime>

#include "init.h"

using namespace std;

int main()
{
  time_t start, end;
  time (&start);
  string pdbfile;
  string pdbfile0,pdbfile2;
  string s,s2;
  string sdir,sdir2;
  int nfile = 1; //number of pdb files, will be determined later
  vector<int> hlist; //list of atom number of hydrogen atoms
  vector< vector<int> > listgrid,listgrid2; // list of atoms in a grid
  vector< vector<int> > listgrid_wat,listgrid_wat2; 
  Mol SYS0,RNA0;
  Hbond PHO;
  ofstream ofile;
  ofile.open("MSD.out",ios::out);
  char oline[2048];

  vector< vector < vector <double> > > r0;
  // loop time interval -- t = k*dt, dt = 10 ps

  for (int k = 0; k < 11; k++) {
    double totSD = 0;
    double ndata = 0;
    int jdir = 1;
    int nframe = 0;

    if (k == 0) {
      stringstream outdir;
      outdir << jdir;
      sdir = outdir.str();
      sdir = "pdb_NPT_pro"+sdir+"/";
// Determine the number of snapshots
      int jfile = 0;
      pdbfile0 = sdir+"1.pdb";
      pdbfile = sdir+"1.pdb";
      while (fexists(pdbfile) == 1) {  // fexists is a self-defined function
        jfile++;
        stringstream out;
        out << jfile+1;
        s = out.str();
        pdbfile = sdir+s+".pdb";
      }
      nfile = jfile;
      r0.resize(nfile);

// Start reading coordinates and calculating mean square displayments, <x(t)-x(0)>, of the hydrogen atoms in RNA for all snapshots

//      if (k == 0) {
      for (int i = -1; i < (nfile - k); i++) {
        if ((i == -1) && (jdir == 1)) {
          Mol RNA,LIG,ION,WAT;
          read_pdb(pdbfile0,SYS0,RNA0,LIG,ION,WAT,listgrid,listgrid_wat);
          for (int j = 0; j < SYS0.iatom.size(); j++) {
            if (SYS0.tr[j] != "TIP" && (SYS0.ta[j].substr(1,1) == "H" || SYS0.ta[j].substr(0,1) == "H")) {
              hlist.push_back(SYS0.iatom[j]);
            }
          }
          for (int ii = 0; ii < nfile; ii++) {
            r0[ii].resize(hlist.size());
            for (int jj = 0; jj < hlist.size(); jj++) {
              r0[ii][jj].resize(3);
            }
          }
        }

                

        if (i >= 0) {
          listgrid.clear();
          listgrid_wat.clear();
          Mol SYS,RNA,LIG,ION,WAT;
          stringstream out;
          out << i+1;
          s = out.str();
          pdbfile = sdir+s+".pdb";

// Calculate mean square displacement
          if (fexists(pdbfile) == 1) {
            read_pdb(pdbfile,SYS,RNA,LIG,ION,WAT,listgrid,listgrid_wat);

            for (int il = 0; il < hlist.size(); il++) {

              for (int kk = 0; kk < 3; kk++) {
                r0[i][il][kk] = SYS.r[hlist[il]-1][kk];
              }
            }
          }
 
        } 
      } // end i
      } // end if k == 0

      if (k > 0) {
        vector<double> r1,r2,r12;
        r1.resize(3);
        r2.resize(3);
        r12.resize(3);
		
        for (int i = 0; i < (nfile - k); i++) {         
          for (int il = 0; il < hlist.size(); il++) {
            for (int kk = 0; kk < 3; kk++) {
              r1[kk] = r0[i][il][kk];
              r2[kk] = r0[i+k][il][kk];
              r12[kk] = r1[kk]-r2[kk];
            }
            totSD = totSD + r12[0]*r12[0]+r12[1]*r12[1]+r12[2]*r12[2];
            ndata = ndata + 1.0;
          }
        }
        double MSD = totSD/ndata;
        sprintf(oline, " %6.3f  %12.8f ", k*0.01, MSD); // time in nanosecond
        ofile << oline << endl;
      } // end if k > 0
  } // end k
  ofile.close();

  time (&end);
  return 0;
}

void read_pdb(string pdbfile, Mol& SYS_read, Mol& RNA_read, Mol& LIG_read, Mol& ION_read, Mol& WAT_read,vector< vector<int> >& listgrid_read, vector< vector<int> >& listgrid_wat_read)
{
  fstream infile;
  infile.open(pdbfile.c_str());
  string line;
  vector<double> origin;
  origin.resize(3);
  origin[0] = 31.16;
  origin[1] = -17.86;
  origin[2] = -2.045;
  double rgrid = 5.0; // The size of grid, in Angstrom
  double rmin = -45.0;
  int ngrid = int((0-rmin)/rgrid)*2;
  listgrid_read.resize(ngrid*ngrid*ngrid);  
  listgrid_wat_read.resize(ngrid*ngrid*ngrid);

  getline(infile,line);
  istringstream ss(line);
  string field;
  double x;
  ss >> field >> x;
  if (field != "CRYST1") {
    cerr << "Error: header is missing!" << endl;
    exit(1);
  }

  SYS_read.lbox = x;


  while (getline(infile,line))
  {
    string str_a;
    str_a = line.substr(0,4);
//    if (str_a == "ATOM")
//    {
      string str_ia;
      string str_ta;
      string str_tr;
      string str_tc;
      string str_ir;
      string str_rx;
      string str_ry;
      string str_rz;
      vector<double> pos;
      vector<int> kgrid;

      str_ia = line.substr(4,7);

      str_ta = line.substr(12,4);
      str_tr = line.substr(17,3);
      str_tc = line.substr(21,1);
      str_ir = line.substr(22,4);

      str_rx = line.substr(30,8);
      str_ry = line.substr(38,8);
      str_rz = line.substr(46,8);
      pos.push_back(atof(str_rx.c_str()));
      pos.push_back(atof(str_ry.c_str()));
      pos.push_back(atof(str_rz.c_str()));
      kgrid.push_back(int((pos[0]-origin[0]-rmin)/rgrid)); // index of grid in 1-D
      kgrid.push_back(int((pos[1]-origin[1]-rmin)/rgrid));
      kgrid.push_back(int((pos[2]-origin[2]-rmin)/rgrid));

      SYS_read.iatom.push_back(atoi(str_ia.c_str()));
      SYS_read.ta.push_back(str_ta);
      SYS_read.tr.push_back(str_tr);
      SYS_read.tc.push_back(str_tc);
      SYS_read.ires.push_back(atoi(str_ir.c_str()));
      SYS_read.r.push_back(pos);
      SYS_read.igrid.push_back(kgrid);

      int kk = kgrid[0]*ngrid*ngrid+kgrid[1]*ngrid+kgrid[2]; // index of grid 
//      cout << "kk = " << kk << endl;
//      listgrid_read[kk].push_back(atoi(str_ia.c_str()));  // store atom in the grid it belongs to 
//      if (str_ta == " OH2" && str_tc == "W") {
//        listgrid_wat_read[kk].push_back(atoi(str_ia.c_str())); // record only atom number of OH2 in water
//      }

      if (str_tc == "R") {
        RNA_read.iatom.push_back(atoi(str_ia.c_str()));
        RNA_read.ta.push_back(str_ta);
        RNA_read.tr.push_back(str_tr);
        RNA_read.tc.push_back(str_tc);
        RNA_read.ires.push_back(atoi(str_ir.c_str()));
        RNA_read.r.push_back(pos);
        RNA_read.igrid.push_back(kgrid);
        listgrid_read[kk].push_back(atoi(str_ia.c_str()));  // store atom in the grid it belongs to
      }

      if (str_tc == "A") {
        listgrid_read[kk].push_back(atoi(str_ia.c_str()));
      }

      if (str_tc == "W") {

        if (str_ta == " OH2") {
          listgrid_wat_read[kk].push_back(atoi(str_ia.c_str())); // record only atom number of OH2 in water
        }

      }
  }

  return;
}

bool fexists(string filename)
{
  ifstream ifile(filename.c_str());
  return ifile;
}

double dist2(vector<double> u1, vector<double> u2)
{
  double distance2 = 0;
  for (int i = 0; i < 3; i++) {
    double x = u1[i]-u2[i];
    distance2 = distance2 + x*x;
  }         
  return distance2;
}

