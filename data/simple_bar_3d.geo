// Gmsh project created on Tue Jan 17 13:21:26 2023
// 0 for triangles, 1 for quads
tri_or_quad = 1;

Point(1) = {0, 0, 0, 1e-2};
Point(2) = {0, 1e-2, 0, 1e-2};
Point(3) = {0, 2e-2, 0, 1e-2};
Point(4) = {1e-2, 0, 0, 1e-2};
Point(5) = {1e-2, 1e-2,0, 1e-2};
Point(6) = {1e-2, 2e-2, 0, 1e-2};


Characteristic Length {:} = 2e-2;

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {4, 5};
Line(4) = {5, 6};
Line(5) = {1, 4};
Line(6) = {2, 5};
Line(7) = {3, 6};

Curve Loop(1) = {5, 3, -6, -1};
Plane Surface(1) = {1};
Curve Loop(2) = {6, 4, -7, -2};
Plane Surface(2) = {2};
Transfinite Surface {1, 2};

If (tri_or_quad == 1)
   Recombine Surface {1, 2};
EndIf

Extrude {0, 0, 1e-2} {
  Surface{1, 2}; Layers{1}; Recombine;
}

Physical Volume(1) = {1, 2};
Physical Surface("right", 11) = {20, 42};
Physical Surface("top", 12) = {46};
Physical Surface("bot", 13) = {16};

Mesh.MshFileVersion = 2.2;
Mesh 3;
