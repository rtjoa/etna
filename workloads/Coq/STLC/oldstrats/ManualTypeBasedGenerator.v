 
From QuickChick Require Import QuickChick. Import QcNotation.
From Coq Require Import Bool ZArith List. Import ListNotations.
From ExtLib Require Import Monad.
From ExtLib.Data.Monads Require Import OptionMonad.
Import MonadNotation.

From STLC Require Import Impl Spec.

Definition manual_gen_typ :=
    fun s : nat =>
    (let
       fix arb_aux (size : nat) : G Typ :=
         match size with
         | 0 => returnGen TBool
         | S size' =>
             freq [ (1, returnGen TBool);
             (1,
             bindGen (arb_aux size')
               (fun p0 : Typ =>
                bindGen (arb_aux size')
                  (fun p1 : Typ => returnGen (TFun p0 p1))))]
         end in
     arb_aux) s.

#[global]
Instance genTyp : GenSized (Typ) := 
  {| arbitrarySized n := manual_gen_typ n |}.

Definition manual_gen_expr :=
  fun s : nat =>
  (let
     fix arb_aux (size : nat) : G Expr :=
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
              bindGen (arb_aux size')
                (fun p1 : Expr => returnGen (Abs p0 p1))));
           (1,
           bindGen (arb_aux size')
             (fun p0 : Expr =>
              bindGen (arb_aux size')
                (fun p1 : Expr => returnGen (App p0 p1))))]
       end in
   arb_aux) s.

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
