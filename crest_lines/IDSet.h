/*
Fast and Robust Detection of Crest Lineson Meshes C++ code
Copyright:(c) Shin Yoshizawa, 2004
E-mail: shin.yoshizawa@mpi-sb.mpg.de
URL: http://www.mpi-sb.mpg.de/~shin
Affiliation: Max-Planck-Institut fuer Informatik: Computer Graphics Group 
 Stuhlsatzenhausweg 85, 66123 Saarbruecken, Germany
 Phone +49 681 9325-408 Fax +49 681 9325-499 

 All right is reserved by Shin Yoshizawa.
This C++ sources are allowed for only primary user of 
research and educational purposes. Don't use secondary: copy, distribution, 
diversion, business purpose, and etc.. 
 */

class IDSet{
  
 public:
  IDSet(){}
  virtual ~IDSet(){}
  void AppendVF(int,IDList *);
  //void AppendVF(int,CotList *);
  
  int SearchI(int dID,IDList *dIHead,IDList *dITail);
  void AppendI(int dID,IDList *dIHead,IDList *dITail,int nowID,int *dnum);
  void AppendI(int dID,IDList *dIHead,IDList *dITail,int nowID);
  
  void AppendIF(int dID,IDList *dIHead,IDList *dITail,int nowID,int *dnum);
  void AppendISort(int dID,IDList *dIHead,IDList *dITail,int nowID,int *dnum);
  void AppendISort(int *dnum,int dID,IDList *dIHead,IDList *dITail,int nowID);
  
  void AppendI(int dID,IDList *dIHead,IDList *dITail);
  void AppendISort(int dID,IDList *dIHead,IDList *dITail,int *num);
  void CleanNeighbor(IDList*,IDList*);
  
  void CleanNeighborLL(IDList **,IDList **,int ,int *);
  void CleanNeighborL(IDList **,IDList **,int );
  //void CleanNeighborL(CotList **,CotList **,int );
  void CleanNeighborL(PolarList **,PolarList **,int );
  void CleanNeighborNot(IDList* dHead,IDList* dTail);
  void Clean(IDList **dFHead,IDList **dFTail,int numberSV,int *dneighborN);
  void AppendICl(int dID,IDList *dIHead,IDList *dITail,int *dnum);
  void AppendPolarI(int dID,PolarList *dITail,double dx,double dy);
 private:
  IDSet(const IDSet& rhs);
  const IDSet &operator=(const IDSet& rhs);

};


