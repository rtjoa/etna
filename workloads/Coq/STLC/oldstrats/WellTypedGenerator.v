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
         | (1, 14) => 439
         | (1, 15) => 522
         | (2, 10) => 209
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
         | (0, 11) => 429
         | (0, 12) => 697
         | (0, 13) => 303
         | _ => 500
         end in
      freq [ (weight_var, bindGen arbitrary (fun p0 : nat => returnGen (Var p0)));
      (1000 - weight_var, bindGen arbitrary (fun p0 : bool => returnGen (Bool p0)))]
  | S size' =>
      let weight_var := match (size,last_callsite) with 
         | (1, 11) => 2
         | (1, 12) => 486
         | (1, 13) => 484
         | (2, 11) => 1
         | (2, 12) => 488
         | (2, 13) => 488
         | (3, 11) => 1
         | (3, 12) => 485
         | (3, 13) => 484
         | (4, 11) => 1
         | (4, 12) => 483
         | (4, 13) => 483
         | (5, 20) => 1
         | _ => 500
         end in
      let weight_boolean := match (size,last_callsite) with 
         | (1, 11) => 200
         | (1, 12) => 481
         | (1, 13) => 535
         | (2, 11) => 26
         | (2, 12) => 488
         | (2, 13) => 529
         | (3, 11) => 10
         | (3, 12) => 484
         | (3, 13) => 536
         | (4, 11) => 4
         | (4, 12) => 483
         | (4, 13) => 546
         | (5, 20) => 2
         | _ => 500
         end in
      let weight_abs := match (size,last_callsite) with 
         | (1, 11) => 939
         | (1, 12) => 536
         | (1, 13) => 494
         | (2, 11) => 950
         | (2, 12) => 533
         | (2, 13) => 493
         | (3, 11) => 952
         | (3, 12) => 545
         | (3, 13) => 495
         | (4, 11) => 954
         | (4, 12) => 549
         | (4, 13) => 484
         | (5, 20) => 956
         | _ => 500
         end in
      let weight_app := match (size,last_callsite) with 
         | (1, 11) => 1
         | (1, 12) => 495
         | (1, 13) => 486
         | (2, 11) => 1
         | (2, 12) => 488
         | (2, 13) => 488
         | (3, 11) => 1
         | (3, 12) => 484
         | (3, 13) => 484
         | (4, 11) => 1
         | (4, 12) => 483
         | (4, 13) => 484
         | (5, 20) => 1
         | _ => 500
         end in
      freq [
      (weight_var,

         let weight_1 := match (size,last_callsite) with 
         | (1, 11) => 521
         | (1, 12) => 496
         | (1, 13) => 498
         | (2, 11) => 457
         | (2, 12) => 500
         | (2, 13) => 500
         | (3, 11) => 503
         | (3, 12) => 498
         | (3, 13) => 500
         | (4, 11) => 449
         | (4, 12) => 500
         | (4, 13) => 500
         | (5, 20) => 500
         | _ => 500
         end in
        bindGen (freq [ (weight_1, returnGen 1); (1000-weight_1, returnGen 0)])
        (fun n1 : nat =>  
         let weight_2 := match (size,last_callsite) with 
         | (1, 11) => 480
         | (1, 12) => 500
         | (1, 13) => 502
         | (2, 11) => 468
         | (2, 12) => 500
         | (2, 13) => 500
         | (3, 11) => 389
         | (3, 12) => 498
         | (3, 13) => 500
         | (4, 11) => 449
         | (4, 12) => 500
         | (4, 13) => 500
         | (5, 20) => 500
         | _ => 500
         end in
        bindGen (freq [ (weight_2, returnGen 2); (1000-weight_2, returnGen 0)])
        (fun n2 : nat =>  
         let weight_4 := match (size,last_callsite) with 
         | (1, 11) => 292
         | (1, 12) => 496
         | (1, 13) => 498
         | (2, 11) => 406
         | (2, 12) => 500
         | (2, 13) => 500
         | (3, 11) => 389
         | (3, 12) => 498
         | (3, 13) => 500
         | (4, 11) => 449
         | (4, 12) => 500
         | (4, 13) => 500
         | (5, 20) => 500
         | _ => 500
         end in
        bindGen (freq [ (weight_4, returnGen 4); (1000-weight_4, returnGen 0)])
        (fun n4 : nat =>  
         let weight_8 := match (size,last_callsite) with 
         | (1, 11) => 292
         | (1, 12) => 496
         | (1, 13) => 498
         | (2, 11) => 406
         | (2, 12) => 500
         | (2, 13) => 500
         | (3, 11) => 389
         | (3, 12) => 498
         | (3, 13) => 500
         | (4, 11) => 449
         | (4, 12) => 500
         | (4, 13) => 500
         | (5, 20) => 500
         | _ => 500
         end in
        bindGen (freq [ (weight_8, returnGen 8); (1000-weight_8, returnGen 0)])
        (fun n8 : nat =>  
         let weight_16 := match (size,last_callsite) with 
         | (1, 11) => 292
         | (1, 12) => 496
         | (1, 13) => 498
         | (2, 11) => 406
         | (2, 12) => 500
         | (2, 13) => 500
         | (3, 11) => 389
         | (3, 12) => 498
         | (3, 13) => 500
         | (4, 11) => 449
         | (4, 12) => 500
         | (4, 13) => 500
         | (5, 20) => 500
         | _ => 500
         end in
        bindGen (freq [ (weight_16, returnGen 16); (1000-weight_16, returnGen 0)])
        (fun n16 : nat =>  
         let weight_32 := match (size,last_callsite) with 
         | (1, 11) => 292
         | (1, 12) => 496
         | (1, 13) => 498
         | (2, 11) => 406
         | (2, 12) => 500
         | (2, 13) => 500
         | (3, 11) => 389
         | (3, 12) => 498
         | (3, 13) => 500
         | (4, 11) => 449
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
         | (1, 11) => 537
         | (1, 12) => 500
         | (1, 13) => 507
         | (2, 11) => 477
         | (2, 12) => 500
         | (2, 13) => 494
         | (3, 11) => 468
         | (3, 12) => 500
         | (3, 13) => 501
         | (4, 11) => 453
         | (4, 12) => 500
         | (4, 13) => 502
         | (5, 20) => 499
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


