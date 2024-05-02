 
From QuickChick Require Import QuickChick. Import QcNotation.
From Coq Require Import Bool ZArith List. Import ListNotations.
From ExtLib Require Import Monad.
From ExtLib.Data.Monads Require Import OptionMonad.
Import MonadNotation.

From STLC Require Import Impl Spec.

Fixpoint manual_gen_typ (size : nat) : G Typ :=
  match size with
  | 0 => returnGen TBool
  | S size' =>
      freq [ (1, returnGen TBool);
      (1,
      bindGen (manual_gen_typ size')
        (fun p0 : Typ =>
         bindGen (manual_gen_typ size')
           (fun p1 : Typ => returnGen (TFun p0 p1))))]
  end.

#[global]
Instance genTyp : GenSized (Typ) := 
  {| arbitrarySized n := manual_gen_typ n |}.

Fixpoint manual_gen_expr (size : nat) : G Expr :=
  match size with
  | 0 =>
      oneOf [bindGen arbitrary (fun p0 : nat => returnGen (Var p0));
      bindGen arbitrary (fun p0 : bool => returnGen (Bool p0))]
  | S size' =>
      freq [ (1,
      bindGen arbitrary (fun p0 : nat => returnGen (Var p0)));
      (1, bindGen arbitrary (fun p0 : bool => returnGen (Bool p0)));
      (1,
      bindGen arbitrary
        (fun p0 : Typ =>
         bindGen (manual_gen_expr size')
           (fun p1 : Expr => returnGen (Abs p0 p1))));
      (1,
      bindGen (manual_gen_expr size')
        (fun p0 : Expr =>
         bindGen (manual_gen_expr size')
           (fun p1 : Expr => returnGen (App p0 p1))))]
  end.

#[global]
Instance genExpr : GenSized (Expr) := 
  {| arbitrarySized n := manual_gen_expr n |}.
  
Definition test_prop_SinglePreserve :=
  forAll arbitrary (fun (e: Expr) =>
    prop_SinglePreserve e).

(*! QuickChick test_prop_SinglePreserve. *)
  
Definition test_prop_MultiPreserve :=
  forAll arbitrary (fun (e: Expr) =>
    prop_MultiPreserve e).
  
(*! QuickChick test_prop_MultiPreserve. *)
