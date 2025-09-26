Include "../../scripts/templates/Lib_Materials.pro"

Group {
  DSV = Region[(matNum+1)];
  Air = Region[(matNum+2)];
  Boundary_Air = Region[(matNum+3)];
  Domain_Mat = Region[{}];
  Domain_Mag = Region[{}];
  Domain_Iron = Region[{}];
  For i In {1:matNum}
    Mat~{i} = Region[i];
    Domain_Mat += Region[Mat~{i}];
    If (mu_r~{i} == 1.05) 
      Domain_Mag += Region[Mat~{i}];
    EndIf
    If (mu_r~{i} == 667.75) 
      Domain_Iron += Region[Mat~{i}];
    EndIf
  EndFor
  Domain_Air = Region[{Air, DSV}];
  Domain_All = Region[{Domain_Air, Domain_Mat}];

  Vol_Mag = Region[ {Domain_Air, Domain_Mag, Domain_Iron} ] ;
  Vol_NL_Mag = Region[ Domain_Iron ] ;
  Vol_M_Mag   = Region[ Domain_Mag ] ;
}

Function {
  mu0 = 4*Pi*1e-7;

  nu[Domain_Air] = 1.0/mu0;
  mu[Domain_Air] = mu0;

  nu[Domain_Mag] = 1.0/(1.05 * mu0);
  mu[Domain_Mag] = 1.05 * mu0;
  //br[Domain_Mag] = Vector[0, 0, 1.4];
  //hc[Domain_Mag] = Vector[0, 0, 907000];
  For i In {1:matNum}
    hc[Mat~{i}] = Rotate[Vector[polarization_x~{i}, polarization_y~{i}, polarization_z~{i}]/1.445*907000, 0, 0, angle~{i}];
  EndFor

  nu[Domain_Iron] = Steel1010_nu[$1] ;
  dhdb[Domain_Iron] = Steel1010_dhdb[$1];
  mu[Domain_Iron] = Steel1010_mu[$1] ;
  dbdh[Domain_Iron] = Steel1010_dbdh[$1];
}

//Constraint {
  //{ Name a ; Type Assign ;
    //Case {
      //{ Region Domain_All ; SubRegion Boundary_Air ; Value 0. ; }
    //}
  //}
  //{ Name phi ; Type Assign ;
    //Case {
      //{ Region Domain_All ; SubRegion Boundary_Air ; Value 0. ; }
    //}
  //}
//}

modelPath = CurrentDirectory;
Include "../../scripts/templates/Lib_Magnetostatics_a_phi.pro"

//Jacobian {
  //{ Name JVol ;
    //Case {
      //{ Region All ; Jacobian Vol ; }
    //}
  //}
//}

//Integration {
  //{ Name I1 ;
    //Case {
      //{ Type Gauss ;
        //Case {
	        //{ GeoElement Triangle ; NumberOfPoints 4 ; }
	        //{ GeoElement Quadrangle  ; NumberOfPoints 4 ; }
          //{ GeoElement Tetrahedron  ; NumberOfPoints 4 ; }
	      //}
      //}
    //}
  //}
//}

Constraint {
  { Name a ;
    Case {
      { Region Boundary_Air ; Value 0. ; }
    }
  }
  { Name GaugeCondition_a ; Type Assign ;
    Case {
      { Region Domain_All ; SubRegion Boundary_Air ; Value 0. ; }
    }
  }
}


//FunctionSpace {
  //{ Name Hcurl; Type Form1;
    //BasisFunction {
      //{ Name se;  NameOfCoef ae;  Function BF_Edge; Support Domain_All ;
        //Entity EdgesOf[ All ]; }
    //}
    //Constraint {
      //{ NameOfCoef ae;  EntityType EdgesOf ; NameOfConstraint a; }
      //{ NameOfCoef ae;  EntityType EdgesOfTreeIn ; EntitySubType StartingOn ;
        //NameOfConstraint GaugeCondition_a ; }
    //}
  //}
//}

//Formulation {
  //{ Name MagSta_a; Type FemEquation ;
    //Quantity {
      //{ Name a  ; Type Local  ; NameOfSpace Hcurl ; }
    //}
    //Equation {
      //Integral { [ nu[] * Dof{d a} , {d a} ] ;
        //In Domain_All ; Jacobian JVol ; Integration I1 ; }
      //Integral { [ nu[] * br[] , {d a} ] ;
        //In Domain_Mat ; Jacobian JVol ; Integration I1 ; }
    //}
  //}
//}

//Resolution {
  //{ Name B;
    //System {
      //{ Name B; NameOfFormulation MagSta_a; }
    //}
    //Operation {
      //Generate[B];
      //Solve[B];
      //SaveSolution[B];
      //PostOperation[Map_b];
    //}
  //}
//}

//PostProcessing {
  //{ Name Magnetostatics_a; NameOfFormulation Magnetostatics_a;
    //Quantity {
      //{ Name bvec; 
        //Value { 
          //Local { [ -{d a} ]; In Domain_All; Jacobian JacVol_Mag; }
        //}
      //}
      //{ Name babs; 
        //Value { 
          //Local { [ Norm[{d a}] ]; In Domain_All; Jacobian JacVol_Mag; }
        //}
      //}
      //{ Name bz; 
        //Value { 
          //Local { [ -CompZ[{d a}] ]; In Domain_All; Jacobian JacVol_Mag; }
        //}
      //}
    //}
  //}
//}

PostOperation {
  { Name B; NameOfPostProcessing Magnetostatics_a;
    Operation {
      Print[ b, OnElementsOf Domain_All, File "b.pos" ];
      Print[ bvec, OnElementsOf Domain_Mag, File "b_mag.pos" ];
      Print[ babs, OnElementsOf Domain_All, File "b_abs.pos" ];
      Print[ bvec, OnElementsOf Domain_Air, File "b_air.pos" ];
      Print[ bz, OnElementsOf DSV, File "dsv_z.pos" ];
      Print[ b, OnElementsOf DSV, File "dsv.pos" ];
      Print[ babs, OnElementsOf Domain_Iron, Format Table, File "iron_abs.txt"];

      {DSV_TEMPLATE}
    }
  }
}