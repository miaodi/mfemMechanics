// Gmsh project created on Tue Apr 04 13:59:05 2023
// 0 for triangles, 1 for quads
tri_or_quad = 1;

Point(1) = {0, 0, 0};
Point(2) = {5, 0, 0};
Point(3) = {10, 0, 0};
Point(4) = {0, 9, 0};
Point(5) = {5, 9, 0};
Point(6) = {10, 9, 0};
Point(7) = {0, 10, 0};
Point(8) = {5, 10, 0};
Point(9) = {10, 10, 0};
// Point(1) = {0, 0, 0, .5};
// Point(2) = {5, 0, 0, .5};
// Point(3) = {10, 0, 0, .5};
// Point(4) = {0, 9, 0, .5};
// Point(5) = {5, 9, 0, .5};
// Point(6) = {10, 9, 0, .5};
// Point(7) = {0, 10, 0, .5};
// Point(8) = {5, 10, 0, .5};
// Point(9) = {10, 10, 0, .5};

Characteristic Length {:} = 2;

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {4, 5};
Line(4) = {5, 6};
Line(5) = {7, 8};
Line(6) = {8, 9};


Line(7) = {1, 4};
Line(8) = {4, 7};
Line(9) = {2, 5};
Line(10) = {5, 8};
Line(11) = {3, 6};
Line(12) = {6, 9};

Curve Loop(1) = {1, 9, -3, -7};
Plane Surface(1) = {1};
Curve Loop(2) = {2, 11, -4, -9};
Plane Surface(2) = {2};
Curve Loop(3) = {3, 10, -5, -8};
Plane Surface(3) = {3};
Curve Loop(4) = {4, 12, -6, -10};
Plane Surface(4) = {4};

Transfinite Surface {1, 2, 3, 4};

If (tri_or_quad == 1)
   Recombine Surface {1, 2, 3, 4};
EndIf

Physical Surface(1) = {1, 2, 3, 4};
Physical Curve("bot", 11) = {1, 2};
Physical Curve("left top", 12) = {8};
Physical Curve("right top", 13) = {12};

Mesh.MshFileVersion = 2.2;
Mesh 2;