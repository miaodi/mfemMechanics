// 0 for tetrahedra, 1 for hexahedra
tet_or_hex = 1;

Point(1) = {0, 0, 0, 1.0};
Point(2) = {10, 0, 0, 1.0};
Point(3) = {10, 1, 0, 1.0};
Point(4) = {0, 1, 0, 1.0};

Characteristic Length {:} = 0.1;

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};
Transfinite Surface {1};

If (tet_or_hex == 1)
   Recombine Surface {1};
   out[] = Extrude {0, 0, .1} { Surface{1}; Layers{1}; Recombine; };
Else
   out[] = Extrude {0, 0, .1} { Surface{1}; Layers{1}; }
EndIf

Physical Volume(1) = {out[1]};
Physical Surface(6) = {1};
Physical Surface(5) = {out[2]};
Physical Surface(4) = {out[4]};
Physical Surface(3) = {out[0]};
Physical Surface(2) = {out[3]};
Physical Surface(1) = {out[5]};

Mesh 3;
Mesh.MshFileVersion = 2.2;
