From QuickChick Require Import QuickChick. Import QcNotation.
From Coq Require Import Bool ZArith List. Import ListNotations.
From ExtLib Require Import Monad.
From ExtLib.Data.Monads Require Import OptionMonad.
Import MonadNotation.

From STLC Require Import Impl Spec.

Fixpoint manual_gen_typ (size : nat) (last_callsite : nat) : G Typ :=
  match size with
  | 0 => returnGen TBool
  | S size' =>
      let weight_tbool := match (size,last_callsite) with 
         | (1, 14) => 753
         | (1, 15) => 241
         | (2, 10) => 178
         | _ => 500
         end in
      freq [ (weight_tbool, returnGen TBool);
      (1000 - weight_tbool,
      bindGen (manual_gen_typ size' 14)
        (fun p0 : Typ =>
         bindGen (manual_gen_typ size' 15)
           (fun p1 : Typ => returnGen (TFun p0 p1))))]
  end.

Fixpoint manual_gen_expr (size : nat) (last_callsite : nat) : G Expr :=
  match size with
  | 0 =>
      let weight_var := match (size,last_callsite) with 
         | (0, 11) => 252
         | (0, 12) => 484
         | (0, 13) => 495
         | _ => 500
         end in
      freq [ (weight_var, bindGen arbitrary (fun p0 : nat => returnGen (Var p0)));
      (1000 - weight_var, bindGen arbitrary (fun p0 : bool => returnGen (Bool p0)))]
  | S size' =>
      let weight_var := match (size,last_callsite) with 
         | (1, 11) => 70
         | (1, 12) => 235
         | (1, 13) => 114
         | (2, 11) => 58
         | (2, 12) => 9
         | (2, 13) => 3
         | (3, 11) => 69
         | (3, 12) => 2
         | (3, 13) => 2
         | (4, 11) => 200
         | (4, 12) => 1
         | (4, 13) => 1
         | (5, 20) => 1
         | _ => 500
         end in
      let weight_boolean := match (size,last_callsite) with 
         | (1, 11) => 52
         | (1, 12) => 124
         | (1, 13) => 74
         | (2, 11) => 54
         | (2, 12) => 9
         | (2, 13) => 8
         | (3, 11) => 74
         | (3, 12) => 2
         | (3, 13) => 2
         | (4, 11) => 201
         | (4, 12) => 1
         | (4, 13) => 1
         | (5, 20) => 1
         | _ => 500
         end in
      let weight_abs := match (size,last_callsite) with 
         | (1, 11) => 894
         | (1, 12) => 850
         | (1, 13) => 902
         | (2, 11) => 626
         | (2, 12) => 243
         | (2, 13) => 331
         | (3, 11) => 498
         | (3, 12) => 24
         | (3, 13) => 24
         | (4, 11) => 549
         | (4, 12) => 3
         | (4, 13) => 3
         | (5, 20) => 1
         | _ => 500
         end in
      let weight_app := match (size,last_callsite) with 
         | (1, 11) => 354
         | (1, 12) => 581
         | (1, 13) => 279
         | (2, 11) => 852
         | (2, 12) => 927
         | (2, 13) => 931
         | (3, 11) => 861
         | (3, 12) => 946
         | (3, 13) => 946
         | (4, 11) => 782
         | (4, 12) => 953
         | (4, 13) => 953
         | (5, 20) => 959
         | _ => 500
         end in
      freq [ (weight_var,
      bindGen arbitrary (fun p0 : nat => returnGen (Var p0)));
      (weight_boolean, bindGen arbitrary (fun p0 : bool => returnGen (Bool p0)));
      (weight_abs,
      bindGen (manual_gen_typ 2 10)
        (fun p0 : Typ =>
         bindGen (manual_gen_expr size' 11)
           (fun p1 : Expr => returnGen (Abs p0 p1))));
      (weight_app,
      bindGen (manual_gen_expr size' 12)
        (fun p0 : Expr =>
         bindGen (manual_gen_expr size' 13)
           (fun p1 : Expr => returnGen (App p0 p1))))]
  end.

Definition gSized :=
  manual_gen_expr 5 20.
  
Definition test_prop_SinglePreserve :=
  forAll gSized (fun (e: Expr) =>
    prop_SinglePreserve e).

(*! QuickChick test_prop_SinglePreserve. *)
  
Definition test_prop_MultiPreserve :=
  forAll gSized (fun (e: Expr) =>
    prop_MultiPreserve e).
  
(*! QuickChick test_prop_MultiPreserve. *)


