#include "TCanvas.h"
#include "TFile.h"
#include "TH2D.h"
#include "TMath.h"
#include "TStyle.h"
#include "TParameter.h"

#include <vector>

using namespace std;

//#include "PlotSensiAsimov_2D_parallel.C"
enum OscParam {
    not_defined = -1,
    sin2213 = 0,
    sin223,
    sin223_bar,
    deltaCP,
    dm2,
    dm2_bar,
    OscParamCount
};

double Rc(double x);
TH2D *Smeared(TH2D* hinp);
TH2D *Smeared(TH2D* hinp, int axis, double smear_error, int nReBin);
std::vector<double> bicubicresize(const std::vector<double>& in_vec, int src_width, int src_height, int Rebin);
double getpixel(const std::vector<double>& in_vec, int src_width, int src_height, int x, int y);
double CatMullRom( double x );
void DeltaChi2Hist(TH2D* hinp);
// methods for bicubic 1D
TH1D *Resize(int nReBin, TH1D *NLL1D);
std::vector<double> bicubicresize(const std::vector<double>& in_vec, int src_width, int Rebin);
double getpixel(const std::vector<double>& in_vec, int src_width, int x);




//###################################################
void SmearingNoShift(char *iFile, char *oFile)
{
  TFile *f=new TFile(iFile,"read");
  TH2D *hNH_org=(TH2D*)f->Get("NLL2D_NH");
  TH2D *hNH_new=Smeared(hNH_org);
  // DeltaChi2Hist(hNH_new);
  TH2D *hIH_org=(TH2D*)f->Get("NLL2D_IH");
  TH2D *hIH_new=Smeared(hIH_org);
  //DeltaChi2Hist(hIH_new);

  TFile *out1=new TFile(oFile,"recreate");
  hNH_new->Write("NLL2D_NH");
  hIH_new->Write("NLL2D_IH");
  out1->Close();
}
//###################################################
void SingleSmearing(char *iFile, char *oFile)
{
  TFile *f=new TFile(iFile,"read");
  TH2D *h_org=(TH2D*)f->Get("DeltaChi2");
  TH2D *h_new=Smeared(h_org);
  DeltaChi2Hist(h_new);

  TFile *out1=new TFile(oFile,"recreate");
  h_new->Write("DeltaChi2");
  out1->Close();
}
//###################################################
void dmSmearing(char *iFile, char *oFile)
{
  TFile *f=new TFile(iFile,"read");
  TH2D *hNH_org=(TH2D*)f->Get("NLL2D_NH");
  TH2D *hNH_new=Smeared(hNH_org);
  DeltaChi2Hist(hNH_new);
  TH2D *hIH_org=(TH2D*)f->Get("NLL2D_IH");
  TH2D *hIH_new=Smeared(hIH_org);
  DeltaChi2Hist(hIH_new);

  TFile *out1=new TFile(oFile,"recreate");
  hNH_new->Write("NLL2D_NH");
  hIH_new->Write("NLL2D_IH");
  out1->Close();
}
//###################################################
// 2019/20: 1.446e-5 [TN396]
void Smear(char *infile, char *outfile, double smear_error_dm2 = 1.446e-5)
{
  TFile *f=new TFile(infile,"read");
  TH1 *c1_org = (TH1 *)f->Get("cont");
  TH2D *cont_org=dynamic_cast<TH2D*>(c1_org);

  int nReBin = 10;
  if (cont_org) {
      // 2d
      TParameter<int> *pXParam = (TParameter<int> *)f->Get("xParam");
      TParameter<int> *pYParam = (TParameter<int> *)f->Get("yParam");

      TH2D *cont_new = cont_org;
      if (pXParam->GetVal() == int(dm2)) {
          cont_new = Smeared(cont_org, 1, smear_error_dm2, nReBin);
          DeltaChi2Hist(cont_new); // subtract the minimum$
      }
      else if (pYParam->GetVal() == int(dm2)) {
          cont_new = Smeared(cont_org, 2, smear_error_dm2, nReBin);
          DeltaChi2Hist(cont_new); // subtract the minimum$
      }

      TFile *out1=new TFile(outfile,"recreate");
      cont_new->Write("cont");
      pXParam->Write();
      pYParam->Write();
      f->Get("xParamName")->Write("xParamName");
      f->Get("yParamName")->Write("yParamName");
      out1->Close();
  }
  else {
      // 1d
      f->ls();
      TParameter<int> *pXParam = (TParameter<int> *)f->Get("xParam");

      TH1D *c1_new = (TH1D *)c1_org;
      // for 1D let's simply smeear by default
      {
          TH1D *c1_smooth = Resize(nReBin, (TH1D *)c1_org);
          // now to be able to use the 2d smearing, we create a 2D histogram with the same contents
          int nx = c1_smooth->GetNbinsX();
          TH2D *c2_smooth = new TH2D("c2_smooth", "", nx, c1_smooth->GetXaxis()->GetXmin(), c1_smooth->GetXaxis()->GetXmax(), 1, 0., 1.);
          for (int ix = 0; ix < nx; ix++) {
              double val = c1_smooth->GetBinContent(ix+1);
              c2_smooth->SetBinContent(ix+1,1,val);
          }
          TH2D *c2_new = Smeared(c2_smooth, 1, smear_error_dm2, 1);
          c1_new = (TH1D *)c2_new->ProjectionX("cont_smeared");
          c1_new->GetXaxis()->SetTitle(c1_org->GetXaxis()->GetTitle());
          c1_new->GetYaxis()->SetTitle(c1_org->GetYaxis()->GetTitle());
      }

      TFile *out1=new TFile(outfile,"recreate");
      c1_new->Write("cont");
      pXParam->Write();
      f->Get("xParamName")->Write("xParamName");
      out1->Close();
  }
}
//###################################################
void DeltaChi2Hist(TH2D* hinp)
{
  double Min=1.e6;
  for (int i=0;i<hinp->GetNbinsX();i++)
    {
      for (int j=0;j<hinp->GetNbinsY();j++)
	{
	  if (hinp->GetBinContent(i+1, j+1)<Min)Min=hinp->GetBinContent(i+1, j+1);
	}
    }
  
  for (int i=0;i<hinp->GetNbinsX();i++)
    {
      for (int j=0;j<hinp->GetNbinsY();j++)
	{
	  hinp->SetBinContent(i+1, j+1,hinp->GetBinContent(i+1, j+1)-Min);
	}
    }
}
//###################################################
void Test(char *iFile, char *hName)
{
  gStyle->SetOptStat(0);
  
  TFile *f=new TFile(iFile,"read");
  TH2D *h_org=(TH2D*)f->Get(hName);
  TH2D *h_new=Smeared(h_org);

  TCanvas *c1=new TCanvas("c1","Original",800,600);
  c1->cd();
  h_org->Draw("COLZ");
  
  TCanvas *c2=new TCanvas("c2","New",800,600);
  c2->cd();
  h_new->Draw("COLZ");
}
//###################################################
TH2D *Smeared(TH2D* hinp) {
  //double smear_error = 0.00003556768; //2017
  //double smear_error = 0.00004083384; //2018
  double smear_error = 1.446e-5;      //2019/20
  int nReBin = 10;
  return Smeared(hinp, 2, smear_error, nReBin);
}

const int kAxisX = 1;
const int kAxisY = 2;

TH2D *Smeared(TH2D* hinp, int axis, double smear_error, int nReBin) // axis is 1:x, 2:y
{
  //Get info from histo
  int nbin_X = hinp->GetNbinsX();
  int nbin_Y = hinp->GetNbinsY();
	
  double Y_low = hinp->GetYaxis()->GetXmin();
  double Y_high = hinp->GetYaxis()->GetXmax();
  double X_low = hinp->GetXaxis()->GetXmin();
  double X_high = hinp->GetXaxis()->GetXmax();

  if (axis != kAxisX && axis != kAxisY) { cerr << "Unknown value passed for axis: " << axis << endl; exit(110); }

  int N = nbin_Y*nReBin;
  int nReBin2 = nReBin;
  int N2 = nbin_Y*nReBin2;

  TH2D *SA_smear = new TH2D("SA_smear", hinp->GetTitle(),nbin_X, X_low, X_high, N2, Y_low, Y_high);
  TH2D *SA_L = new TH2D("SA_L", "FakeData Sensitivity", nbin_X, X_low, X_high, N, Y_low, Y_high);

  SA_smear->GetXaxis()->SetTitle(hinp->GetXaxis()->GetTitle());
  SA_smear->GetYaxis()->SetTitle(hinp->GetYaxis()->GetTitle());

  //**** BiCubic resize of the TH2 ****
  TH2D *SA_smoothed;
  if (nReBin > 1) {
      std::vector<double> in_vec2(nbin_X * nbin_Y);
      for (int i=0;i<nbin_X;i++)
        for (int j=0;j<nbin_Y;j++)
          {
        {
          in_vec2[i*nbin_Y+j]=hinp->GetBinContent(i+1,j+1);
        }      
          }
      
      std::vector<double> out_vec2=bicubicresize(in_vec2, nbin_X , nbin_Y , nReBin2);
      SA_smoothed=new TH2D("Smoothed","Test interpolation",nReBin2*hinp->GetNbinsX(),hinp->GetXaxis()->GetXmin(),hinp->GetXaxis()->GetXmax(),nReBin2*hinp->GetNbinsY(),hinp->GetYaxis()->GetXmin(),hinp->GetYaxis()->GetXmax());
      
      for (int i=0;i<nReBin*hinp->GetNbinsX();i++)
        for (int j=0;j<nReBin*hinp->GetNbinsY();j++)
          {
        {
          SA_smoothed->SetBinContent(i+1,j+1, out_vec2.at(i*nReBin2*hinp->GetNbinsY()+j));
        }
          }
  }
  else {
      SA_smoothed = hinp;
  }

  //**** Fill SA_smear with the likelihood in each bin ****
  for(int i=0; i<SA_smear->GetNbinsX(); i++)
    {
      for(int j=0; j<SA_smear->GetNbinsY(); j++)
	{
	  double x_smear_a = SA_smear->GetXaxis()->GetBinCenter(i+1);
	  double y_smear_a = SA_smear->GetYaxis()->GetBinCenter(j+1);
	  int bin_a = SA_smoothed->FindBin(x_smear_a,y_smear_a);
	  double log_l_a = SA_smoothed->GetBinContent(bin_a);
	  double L_a = exp(-0.5*log_l_a);
	  SA_L->SetBinContent(i+1,j+1,L_a);
	}
    }

  //**** Compute factors for gaussian spread
  TAxis *smearAxis = (axis == kAxisY ? SA_smear->GetYaxis() : SA_smear->GetXaxis());
  TAxis *otherAxis = (axis == kAxisY ? SA_smear->GetXaxis() : SA_smear->GetYaxis());
  double *frac=new double[smearAxis->GetNbins()];
  int nBins5sigma=5.*smear_error/smearAxis->GetBinWidth(1);
  double Tot=0;
  for (int i=0; i<TMath::Max(nBins5sigma, smearAxis->GetNbins());i++)
    {
      frac[i]=TMath::Gaus(i*smearAxis->GetBinWidth(1), 0,  smear_error, 0);     
    }
  

  
  //**** Do actual smearing  ****
  for(int i=0; i<otherAxis->GetNbins(); i++) // center of bin which is kept
    {
      double other_center_a = otherAxis->GetBinCenter(i+1);
      for(int j=0; j<smearAxis->GetNbins(); j++)  // center of bin on axis were gaussian is built
	{
      int ix = (axis == kAxisY ? i : j);
      int iy = (axis == kAxisY ? j : i);
	  double L_a= SA_L->GetBinContent(ix+1,iy+1);
	 
	  for(int k=0; k<smearAxis->GetNbins(); k++)
	    {
	      double smear_center_temp_a= smearAxis->GetBinCenter(k+1);
	      int shift=TMath::Abs(j-k);
        double x_temp_a = (axis == kAxisX ? smear_center_temp_a : other_center_a);
        double y_temp_a = (axis == kAxisY ? smear_center_temp_a : other_center_a);
	      SA_smear->Fill(x_temp_a, y_temp_a, frac[shift]*L_a);
	    }
	}
    }
  
  //**** Go back to log space  ****
  for(int i=0; i<SA_smear->GetNbinsX(); i++)
    {
      for(int j=0; j<SA_smear->GetNbinsY(); j++)
	{
	  double logL_a = -2*log(SA_smear->GetBinContent(i+1,j+1));
	  SA_smear->SetBinContent(i+1,j+1,logL_a);	 
	}
    }	
  
  
  return(SA_smear);
}
//###################################################
std::vector<double> bicubicresize(const std::vector<double>& in_vec, int src_width, int src_height, int Rebin)
{
  int dest_width=src_width*Rebin;
  int dest_height=src_height*Rebin;

  std::cout<<"Original number of bins: x=" <<src_width<<", y="<<src_height<<std::endl;
  std::cout<<"Target number of bins: x=" <<dest_width<<", y="<<dest_height<<std::endl;
 
  std::vector<double> out(dest_width * dest_height);

  const double tx=1./((double)Rebin);
  const double ty=1./((double)Rebin);

  std::cout << tx<<" "<<ty<<std::endl;

  for (int i=0;i<dest_width;i++)
    {
      int p=int((i-(double)Rebin/2.)*tx);
      double a=(i-(double)Rebin/2.)*tx-p;
      for (int j=0;j<dest_height;j++)
	{
	  int q=int((j-(double)Rebin/2.)*ty);
	  double b=(j-(double)Rebin/2.)*ty-q;
	    
	  double Val=0;
	  int OK=1;
	  for (int m=-1;m<3;m++)
	    {
	      if (OK==0)break;
	      for (int n=-1;n<3;n++)
		{
		  if (OK==0)break;
		  double ashift=-((double)m-a);
		  double bshift=((double)n-b);
		  // Val+=getpixel(in_vec,src_width,src_height,p+m,q+n)*Rc(ashift)*Rc(bshift);
		  Val+=getpixel(in_vec,src_width,src_height,p+m,q+n)*CatMullRom(ashift)*CatMullRom(bshift);
		  if (Val>1e6)
		    {
		      std::cout<<getpixel(in_vec,src_width,src_height,p+m,q+n)<<" "<<p+m<<" "<<q+n<<std::endl;
		      std::cout<<ashift<<" "<<Rc(ashift)<<" "<<bshift<<" "<<Rc(bshift)<<std::endl;
		      OK=0;
		    }
		}
	    }
	  out[i*dest_height+j]=Val;
	  //  if (Val>0)std::cout<<i<<" "<<j<<" "<<Val<<std::endl;
	}
    }
  return out;
}
//###################################################
double getpixel(const std::vector<double>& in_vec, int src_width, int src_height, int x, int y)
{
  if (x>=0 && y>=0 && x < src_width && y < src_height)
    return in_vec[x*src_height + y];
   
  else if (x<0 && y>=0 && y< src_height) return in_vec[y];
  else if (x<0 && y>= src_height) return in_vec[src_height-1];
  else if (x<0 && y< 0) return in_vec[0];
  else if (x>=src_width && y>=0 && y< src_height) return in_vec[(src_width-1)*src_height+y];
  else if (x>=src_width && y>= src_height) return in_vec[src_width*src_height-1];
  else if (x>=src_width && y<0)return in_vec[(src_width-1)*src_height];
  else if (x>=0 && x < src_width && y<0)return in_vec[x*src_height];
  else if (x>=0 && x < src_width && y>=src_height)return in_vec[x*src_height+src_height-1];
   
  return 0;
}
//###################################################
double CatMullRom( double x )
{
  const double B = 0.0;
  const double C = 0.5;
  double f = x;
  if( f < 0.0 )
    {
      f = -f;
    }
  if( f < 1.0 )
    {
      return ( ( 12 - 9 * B - 6 * C ) * ( f * f * f ) +
	       ( -18 + 12 * B + 6 *C ) * ( f * f ) +
	       ( 6 - 2 * B ) ) / 6.0;
    }
  else if( f >= 1.0 && f < 2.0 )
    {
      return ( ( -B - 6 * C ) * ( f * f * f )
	       + ( 6 * B + 30 * C ) * ( f *f ) +
	       ( - ( 12 * B ) - 48 * C  ) * f +
	       8 * B + 24 * C)/ 6.0;
    }
  else
    {
      return 0.0;
    }
} 
//###################################################
double Rc(double x)
{
  double f = x;
  if( f < 0.0 )
    {
      f = -f;
    }
  
  if( f >= 0.0 && f <= 1.0 )
    {
      return ( 2.0 / 3.0 ) + ( 0.5 ) * ( f* f * f ) - (f*f);
    }
  else if( f > 1.0 && f <= 2.0 )
    {
      return 1.0 / 6.0 *TMath::Power( ( 2.0 - f  ), 3.0 );
    }
  return 1.0; 
}
//###################################################
// the following methods are originally from Bicubic_1D
//###################################################
TH1D *Resize(int nReBin, TH1D *NLL1D)
{
  std::vector<double> in_vec(NLL1D->GetNbinsX());
  for (int i=0;i<NLL1D->GetNbinsX();i++)
    {
      in_vec[i]=NLL1D->GetBinContent(i+1);
    }      
  std::vector<double> out_vec=bicubicresize(in_vec, NLL1D->GetNbinsX(), nReBin);
 
  TH1D * Smoothed=new TH1D("Smoothed","Test interpolation",nReBin*NLL1D->GetNbinsX(),NLL1D->GetXaxis()->GetXmin(),NLL1D->GetXaxis()->GetXmax());
  
  for (int i=0;i<nReBin*NLL1D->GetNbinsX();i++)
    {
      Smoothed->SetBinContent(i+1, out_vec.at(i));
    }

  return Smoothed;
}
//###################################################
std::vector<double> bicubicresize(const std::vector<double>& in_vec, int src_width, int Rebin)
{
  int dest_width=src_width*Rebin;

  std::cout<<"Original number of bins: x=" <<src_width<<std::endl;
  std::cout<<"Target number of bins: x=" <<dest_width<<std::endl;
 
  std::vector<double> out(dest_width);

  const double tx=1./((double)Rebin);

  std::cout << tx<<std::endl;

  for (int i=0;i<dest_width;i++)
    {
      int p=int((i-(double)Rebin/2.)*tx);
      double a=(i-(double)Rebin/2.)*tx-p;
      double Val=0;
      int OK=1;
      for (int m=-1;m<3;m++)
	    {
	      if (OK==0)break;
	      double ashift=-((double)m-a);
	      // Val+=getpixel(in_vec,src_width,src_height,p+m,q+n)*Rc(ashift)*Rc(bshift);
	      Val+=getpixel(in_vec,src_width,p+m)*CatMullRom(ashift);
	      if (Val>1e6)
		{
		  std::cout<<getpixel(in_vec,src_width,p+m)<<" "<<p+m<<std::endl;
		  std::cout<<ashift<<" "<<Rc(ashift)<<std::endl;
		  OK=0;
		}
	    }
      out[i]=Val;
      //  if (Val>0)std::cout<<i<<" "<<j<<" "<<Val<<std::endl;
    }
  return out;
}
//###################################################
double getpixel(const std::vector<double>& in_vec, int src_width, int x)
{
  if (x>=0 && x < src_width) return in_vec[x];
  else if (x<0) return in_vec[0];
  else if (x>=src_width) return in_vec[src_width-1];

  return 0;
}
//###################################################

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_file> [smear_factor]" << std::endl;
        return 1;
    }

    char *infile = argv[1];
    char *outfile = argv[2];

    // Default smear factor
    double smear_factor = 1.446e-5;

    // If the user provides a smear factor, parse it
    if (argc >= 4) {
        try {
            smear_factor = std::stod(argv[3]);
        } catch (const std::exception &e) {
            std::cerr << "Error: Invalid smear factor. Using default value " << smear_factor << std::endl;
        }
    }

    // Call the Smear function
    Smear(infile, outfile, smear_factor);

    return 0;
}
