Group {
  DSV = Region[(2*matNum+1)];
  Air = Region[(2*matNum+2)];
  Boundary_Air = Region[(2*matNum+3)];
  Domain_Mat = Region[{}];
  For i In {1:matNum}
    Mat~{i} = Region[2*i];
    Boundary_Mat~{i} = Region[2*i+1];
    Domain_Mat += Region[Mat~{i}];
  EndFor
  Domain_Air = Region[{Air, DSV}];
  Domain_All = Region[{Domain_Air, Domain_Mat}];
}

Function {
  mu0 = 4*Pi*1e-7;

  nu[Domain_Air] = 1.0/mu0;

  For i In {1:matNum}
    br[Mat~{i}] = Rotate[Vector[0, 0, polarization~{i}], 0, 0, angle~{i}];
    nu[Mat~{i}] = 1.0/(mu_r~{i} * mu0);
  EndFor
}


Jacobian {
  { Name JVol ;
    Case {
      { Region All ; Jacobian Vol ; }
    }
  }
}

Integration {
  { Name I1 ;
    Case {
      { Type Gauss ;
        Case {
	        { GeoElement Triangle ; NumberOfPoints 4 ; }
	        { GeoElement Quadrangle  ; NumberOfPoints 4 ; }
          { GeoElement Tetrahedron  ; NumberOfPoints 4 ; }
	      }
      }
    }
  }
}

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


FunctionSpace {
  { Name Hcurl; Type Form1;
    BasisFunction {
      { Name se;  NameOfCoef ae;  Function BF_Edge; Support Domain_All ;
        Entity EdgesOf[ All ]; }
    }
    Constraint {
      { NameOfCoef ae;  EntityType EdgesOf ; NameOfConstraint a; }
      { NameOfCoef ae;  EntityType EdgesOfTreeIn ; EntitySubType StartingOn ;
        NameOfConstraint GaugeCondition_a ; }
    }
  }
}

Formulation {
  { Name MagSta_a; Type FemEquation ;
    Quantity {
      { Name a  ; Type Local  ; NameOfSpace Hcurl ; }
    }
    Equation {
      Integral { [ nu[] * Dof{d a} , {d a} ] ;
        In Domain_All ; Jacobian JVol ; Integration I1 ; }
      Integral { [ nu[] * br[] , {d a} ] ;
        In Domain_Mat ; Jacobian JVol ; Integration I1 ; }
    }
  }
}

Resolution {
  { Name B;
    System {
      { Name B; NameOfFormulation MagSta_a; }
    }
    Operation {
      Generate[B]; Solve[B]; SaveSolution[B];
      PostOperation[Map_b];
    }
  }
}

PostProcessing {
  { Name B; NameOfFormulation MagSta_a;
    Quantity {
      { Name bvec; 
        Value { 
          Local { [ -{d a} ]; In Domain_All; Jacobian JVol; }
        }
      }
      { Name bx; 
        Value { 
          Local { [ -CompX[{d a}] ]; In Domain_All; Jacobian JVol; Integration I1; }
        }
      }
      { Name by; 
        Value { 
          Local { [ -CompY[{d a}] ]; In Domain_All; Jacobian JVol; Integration I1; }
        }
      }
      { Name bz; 
        Value { 
          Local { [ -CompZ[{d a}] ]; In Domain_All; Jacobian JVol; Integration I1; }
        }
      }
    }
  }
}

PostOperation {
  { Name Map_b; NameOfPostProcessing B;
    Operation {
      Print[ bvec, OnElementsOf Domain_All, File "b.pos" ];
      Print[ bz, OnElementsOf DSV, File "dsv_z.pos" ];
      Print[ bvec, OnElementsOf DSV, File "dsv.pos" ];

      {DSV_TEMPLATE}
    }
  }
}