// Gmsh project created on Tue Jan 17 13:21:26 2023
// 0 for triangles, 1 for quads
tri_or_quad = 1;
thick =1.5e-3;
dt = 1.5e-3;

Point(1) = {0, -1.5e-3, 0, 1.5e-3};
Point(2) = {0, -1e-4, 0, 1.5e-3};
Point(3) = {0, 1e-4, 0, 1.5e-3};
Point(4) = {0, 1.5e-3, 0, 1.5e-3};
Point(5) = {100e-3, -1.5e-3, 0, 1.5e-3};
Point(6) = {100e-3, 1.5e-3, 0, 1.5e-3};
Point(7) = {30e-3, 0, 0, 1.5e-3};
Point(8) = {100e-3, 0, 0, 1.5e-3};
Point(9) = {30e-3, 1.5e-3, 0, 1.5e-3};
Point(10) = {30e-3, -1.5e-3, 0, 1.5e-3};

Characteristic Length {:} = 1.5e-3;

Line(1) = {1, 10};
Line(2) = {5, 8};
Line(3) = {6, 9};
Line(4) = {4, 3};
Line(5) = {3, 7};
Line(6) = {7, 2};
Line(7) = {2, 1};

Line(8) = {10, 7};
Line(9) = {7, 9};
Line(10) = {7, 8};
Line(11) = {9, 4};
Line(12) = {10, 5};
Line(13) = {8, 6};

Curve Loop(1) = {1, 8, 6, 7};
Plane Surface(1) = {1};
Transfinite Surface {1};
Recombine Surface {1};
Curve Loop(2) = {12, 2, -10, -8};
Plane Surface(2) = {2};
Transfinite Surface {2};
Recombine Surface {2};
Curve Loop(3) = {10, 13, 3, -9};
Plane Surface(3) = {3};
Transfinite Surface {3};
Recombine Surface {3};
Curve Loop(4) = {5, 9, 11, 4};
Plane Surface(4) = {4};
Transfinite Surface {4};
Recombine Surface {4};

Extrude {0, 0, thick} {
  Surface{1}; Layers{2}; Recombine;
}
Extrude {0, 0, thick} {
  Surface{2}; Layers{2}; Recombine;
}
Extrude {0, 0, thick} {
  Surface{3}; Layers{2}; Recombine;
}
Extrude {0, 0, thick} {
  Surface{4}; Layers{2}; Recombine;
}

Physical Volume(1) = {1, 2, 3, 4};
Physical Surface("right", 11) = {48, 70};
Physical Surface("left top", 12) = {100};
Physical Surface("left bot", 13) = {34};
Mesh.MshFileVersion = 2.2;
Mesh 3;
