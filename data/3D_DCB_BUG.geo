// Gmsh project created on Tue Jan 17 13:21:26 2023
// 0 for triangles, 1 for quads
tri_or_quad = 1;
thick = 0.1;

Point(1)   ={0       ,-0.15,0};
Point(2)   ={0       ,-0.05,0};
Point(3)   ={0       ,0    ,0};
Point(4)   ={0       ,0.05 ,0};
Point(5)   ={0       ,.15 ,0};
Point(6)   ={0.25    ,-0.15,0};
Point(7)   ={0.25    ,-0.05,0};
Point(8)   ={0.25    ,0    ,0};
Point(9)   ={0.25    ,0.05 ,0};
Point(10)  ={0.25    ,.15  ,0};
Point(11)  ={0.5     ,-0.15,0};
Point(12)  ={0.5     ,-0.05,0};
Point(13)  ={0.5     ,0.05 ,0};
Point(14)  ={0.5     ,0.15 ,0};


Line(1)  ={1 ,2  };
Line(2)  ={2 ,3  };
Line(3)  ={3 ,4  };
Line(4)  ={4 ,5  };
Line(5)  ={6 ,7  };
Line(6)  ={7 ,8  };
Line(7)  ={8 ,9  };
Line(8)  ={9 ,10 };
Line(9)  ={11,12 };
Line(10) ={13,14 };


Line(11) ={1 ,6  };
Line(12) ={6 ,11 };
Line(13) ={2 ,7  };
Line(14) ={7 ,12 };
Line(15) ={3 ,8  };
Line(16) ={4 ,9  };
Line(17) ={9 ,13 };
Line(18) ={5 ,10 };
Line(19) ={10,14 };


Characteristic Length {:} = .5;

Curve Loop(1) = {11, 5, -13, -1};
Plane Surface(1) = {1};
Transfinite Surface {1};
Recombine Surface {1};
Extrude {0, 0, thick} {
  Surface{1}; Layers{1}; Recombine;
}

Curve Loop(2) = {13, 6, -15,-2};
Plane Surface(2) = {2};
Transfinite Surface {2};
Recombine Surface {2};
Extrude {0, 0, thick} {
  Surface{2}; Layers{1}; Recombine;
}

Curve Loop(3) = {15, 7, -16, -3};
Plane Surface(3) = {3};
Transfinite Surface {3};
Recombine Surface {3};
Extrude {0, 0, thick} {
  Surface{3}; Layers{1}; Recombine;
}


Curve Loop(4) = {16, 8, -18,-4};
Plane Surface(4) = {4};
Transfinite Surface {4};
Recombine Surface {4};
Extrude {0, 0, thick} {
  Surface{4}; Layers{1}; Recombine;
}

Curve Loop(5) = {12, 9, -14, -5};
Plane Surface(5) = {5};
Transfinite Surface {5};
Recombine Surface {5};
Extrude {0, 0, thick} {
  Surface{5}; Layers{1}; Recombine;
}

Curve Loop(6) = {17, 10, -19,-8};
Plane Surface(6) = {6};
Transfinite Surface {6};
Recombine Surface {6};
Extrude {0, 0, thick} {
  Surface{6}; Layers{1}; Recombine;
}

Physical Volume(1) = {1, 2, 3, 4, 5, 6};
Physical Surface("CT", 11) = {40, 62, 84,106};
Physical Surface("CBL", 12) = {120};
Physical Surface("CBR", 13) = {142};
Mesh.MshFileVersion = 2.2;
Mesh 3;