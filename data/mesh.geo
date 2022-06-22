// 0 for triangles, 1 for quads
tri_or_quad = 1;
dt = 0.5 ;
Point(1) = {0, 0, 0, dt};
Point(2) = {1, 0, 0, dt};
Point(3) = {1, 1, 0, dt};
Point(4) = {0, 1, 0, dt};


Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};
Transfinite Surface {1};

If (tri_or_quad == 1)
   Recombine Surface {1};
EndIf

Physical Surface("surf") = {1};
Physical Curve("left") = {4};
Physical Curve("right") = {2};

Mesh.MshFileVersion = 2.2;
Mesh 2;
