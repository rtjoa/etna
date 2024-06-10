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
         | (1, 14) => 500
         | (1, 15) => 500
         | (2, 10) => 500
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
         | (0, 11) => 500
         | (0, 12) => 500
         | (0, 13) => 500
         | _ => 500
         end in
      freq [ (weight_var, bindGen arbitrary (fun p0 : nat => returnGen (Var p0)));
      (1000 - weight_var, bindGen arbitrary (fun p0 : bool => returnGen (Bool p0)))]
  | S size' =>
      let weight_var := match (size,last_callsite) with 
         | (1, 11) => 500
         | (1, 12) => 500
         | (1, 13) => 500
         | (2, 11) => 500
         | (2, 12) => 500
         | (2, 13) => 500
         | (3, 11) => 500
         | (3, 12) => 500
         | (3, 13) => 500
         | (4, 11) => 500
         | (4, 12) => 500
         | (4, 13) => 500
         | (5, 20) => 500
         | _ => 500
         end in
      let weight_boolean := match (size,last_callsite) with 
         | (1, 11) => 500
         | (1, 12) => 500
         | (1, 13) => 500
         | (2, 11) => 500
         | (2, 12) => 500
         | (2, 13) => 500
         | (3, 11) => 500
         | (3, 12) => 500
         | (3, 13) => 500
         | (4, 11) => 500
         | (4, 12) => 500
         | (4, 13) => 500
         | (5, 20) => 500
         | _ => 500
         end in
      let weight_abs := match (size,last_callsite) with 
         | (1, 11) => 500
         | (1, 12) => 500
         | (1, 13) => 500
         | (2, 11) => 500
         | (2, 12) => 500
         | (2, 13) => 500
         | (3, 11) => 500
         | (3, 12) => 500
         | (3, 13) => 500
         | (4, 11) => 500
         | (4, 12) => 500
         | (4, 13) => 500
         | (5, 20) => 500
         | _ => 500
         end in
      let weight_app := match (size,last_callsite) with 
         | (1, 11) => 500
         | (1, 12) => 500
         | (1, 13) => 500
         | (2, 11) => 500
         | (2, 12) => 500
         | (2, 13) => 500
         | (3, 11) => 500
         | (3, 12) => 500
         | (3, 13) => 500
         | (4, 11) => 500
         | (4, 12) => 500
         | (4, 13) => 500
         | (5, 20) => 500
         | _ => 500
         end in
      freq [
      (weight_var,

         let weight_1 := match (size,last_callsite) with 
         | (1, 11) => 500
         | (1, 12) => 500
         | (1, 13) => 500
         | (2, 11) => 500
         | (2, 12) => 500
         | (2, 13) => 500
         | (3, 11) => 500
         | (3, 12) => 500
         | (3, 13) => 500
         | (4, 11) => 500
         | (4, 12) => 500
         | (4, 13) => 500
         | (5, 20) => 500
         | _ => 500
         end in
        bindGen (freq [ (weight_1, returnGen 1); (1000-weight_1, returnGen 0)])
        (fun n1 : nat =>  
         let weight_2 := match (size,last_callsite) with 
         | (1, 11) => 500
         | (1, 12) => 500
         | (1, 13) => 500
         | (2, 11) => 500
         | (2, 12) => 500
         | (2, 13) => 500
         | (3, 11) => 500
         | (3, 12) => 500
         | (3, 13) => 500
         | (4, 11) => 500
         | (4, 12) => 500
         | (4, 13) => 500
         | (5, 20) => 500
         | _ => 500
         end in
        bindGen (freq [ (weight_2, returnGen 2); (1000-weight_2, returnGen 0)])
        (fun n2 : nat =>  
         let weight_4 := match (size,last_callsite) with 
         | (1, 11) => 500
         | (1, 12) => 500
         | (1, 13) => 500
         | (2, 11) => 500
         | (2, 12) => 500
         | (2, 13) => 500
         | (3, 11) => 500
         | (3, 12) => 500
         | (3, 13) => 500
         | (4, 11) => 500
         | (4, 12) => 500
         | (4, 13) => 500
         | (5, 20) => 500
         | _ => 500
         end in
        bindGen (freq [ (weight_4, returnGen 4); (1000-weight_4, returnGen 0)])
        (fun n4 : nat =>  
         let weight_8 := match (size,last_callsite) with 
         | (1, 11) => 500
         | (1, 12) => 500
         | (1, 13) => 500
         | (2, 11) => 500
         | (2, 12) => 500
         | (2, 13) => 500
         | (3, 11) => 500
         | (3, 12) => 500
         | (3, 13) => 500
         | (4, 11) => 500
         | (4, 12) => 500
         | (4, 13) => 500
         | (5, 20) => 500
         | _ => 500
         end in
        bindGen (freq [ (weight_8, returnGen 8); (1000-weight_8, returnGen 0)])
        (fun n8 : nat =>  
         let weight_16 := match (size,last_callsite) with 
         | (1, 11) => 500
         | (1, 12) => 500
         | (1, 13) => 500
         | (2, 11) => 500
         | (2, 12) => 500
         | (2, 13) => 500
         | (3, 11) => 500
         | (3, 12) => 500
         | (3, 13) => 500
         | (4, 11) => 500
         | (4, 12) => 500
         | (4, 13) => 500
         | (5, 20) => 500
         | _ => 500
         end in
        bindGen (freq [ (weight_16, returnGen 16); (1000-weight_16, returnGen 0)])
        (fun n16 : nat =>  
         let weight_32 := match (size,last_callsite) with 
         | (1, 11) => 500
         | (1, 12) => 500
         | (1, 13) => 500
         | (2, 11) => 500
         | (2, 12) => 500
         | (2, 13) => 500
         | (3, 11) => 500
         | (3, 12) => 500
         | (3, 13) => 500
         | (4, 11) => 500
         | (4, 12) => 500
         | (4, 13) => 500
         | (5, 20) => 500
         | _ => 500
         end in
        bindGen (freq [ (weight_32, returnGen 32); (1000-weight_32, returnGen 0)])
        (fun n32 : nat =>  
        let p1 := n1+n2+n4+n8+n16+n32 in
        returnGen (Var p1))
      ))))));
      (weight_boolean,
        let weight_true := match (size,last_callsite) with 
         | (1, 11) => 500
         | (1, 12) => 500
         | (1, 13) => 500
         | (2, 11) => 500
         | (2, 12) => 500
         | (2, 13) => 500
         | (3, 11) => 500
         | (3, 12) => 500
         | (3, 13) => 500
         | (4, 11) => 500
         | (4, 12) => 500
         | (4, 13) => 500
         | (5, 20) => 500
         | _ => 500
         end in
        freq [ (weight_true, returnGen (Bool true)); (1000 - weight_true, returnGen (Bool false))]
      );
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


