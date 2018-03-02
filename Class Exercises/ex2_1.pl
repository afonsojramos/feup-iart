% 2.1

:-use_module(library(lists)).
%estadoinicial
estado_inicial(b(0,0)).

%estadofinal
estado_final(b(2,0)).

%transicoesEntreEstados
sucessor(b(X,Y), b(4,Y)) :- X<4. %encher balde X
sucessor(b(X,Y), b(X,3)) :- Y<3. %encher balde Y
sucessor(b(X,Y), b(0,Y)) :- X>0. %esvaziar balde X
sucessor(b(X,Y), b(X,0)) :- Y>0. %esvaziar balde Y
sucessor(b(X,Y), b(4,Y1)) :- %encher balde X com parte de Y
			X+Y>=4,
			X<4,
			Y1 is Y-(4-X).
sucessor(b(X,Y), b(X1,3)) :- %encher balde Y com parte de X
			X+Y>=3,
			Y<3,
			X1 is X-(3-Y).
sucessor(b(X,Y), b(X1,0)) :- %esvaziar balde Y para X
			X+Y<4,
			Y>0,
			X1 is X+Y.
sucessor(b(X,Y), b(0,Y1)) :- %esvaziar balde X para Y
			X+Y<3,
			X>0,
			Y1 is X+Y.

solvedfs(S):-
	estado_inicial(Ei),
	dfs(Ei, [Ei], S).

dfs(E,_,[E]):-estado_final(E).
dfs(E, V, [E,R]):-
	sucessor(E,E2), \+member(E2,V),
	dfs(E2,[E2|V],R).


solve_bfs(S):-
	estado_inicial(Ei),
	bfs([[Ei]], SR),
	reverse(SR,S).

bfs([[E|Path]|_], [E|Path]):-estado_final(E).
bfs([[E|Path]|R], S):-
	findall([E2|[E|Path]], sucessor(E,E2), LS),
	append(R,LS,L2),
	bfs(L2,S).
