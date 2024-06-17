From QuickChick Require Import QuickChick. Import QcNotation.
From Coq Require Import Bool ZArith List. Import ListNotations.
From ExtLib Require Import Monad.
From ExtLib.Data.Monads Require Import OptionMonad.
Import MonadNotation.

From STLC Require Import Impl Spec.

Fixpoint genTyp (size : nat) (stack1 : nat) (stack2 : nat) : G (Typ) :=
  match size with
  | O  => 
    (* Frequency1 (single-branch) *) 
    (returnGen (TBool ))
  | S size1 => 
    (* Frequency2 *) (freq [
      (* TBool *) (match (size, stack1, stack2) with
      | (1, 2, 1) => 50
      | (1, 2, 4) => 50
      | (2, 0, 2) => 50
      | (2, 3, 2) => 50
      | (2, 5, 2) => 50
      | (2, 6, 2) => 50
      | _ => 500
      end,
      (returnGen (TBool ))); 
      (* TFun *) (match (size, stack1, stack2) with
      | (1, 2, 1) => 50
      | (1, 2, 4) => 50
      | (2, 0, 2) => 50
      | (2, 3, 2) => 50
      | (2, 5, 2) => 50
      | (2, 6, 2) => 50
      | _ => 500
      end,
      (bindGen (genTyp size1 stack2 1) 
      (fun p1 => 
        (bindGen (genTyp size1 stack2 4) 
        (fun p2 => 
          (returnGen (TFun p1 p2)))))))])
  end.

Fixpoint genExpr (size : nat) (stack1 : nat) (stack2 : nat) : G (Expr) :=
  match size with
  | O  => 
    (* Frequency3 *) (freq [
      (* Var *) (match (size, stack1, stack2) with
      | (0, 3, 3) => 95
      | (0, 3, 5) => 78
      | (0, 3, 6) => 71
      | (0, 5, 3) => 95
      | (0, 5, 5) => 56
      | (0, 5, 6) => 63
      | (0, 6, 3) => 95
      | (0, 6, 5) => 80
      | (0, 6, 6) => 49
      | _ => 500
      end,
      (bindGen 
      (* GenNat1 *)
      (let _weight_1 := match (size, stack1, stack2) with
      | (0, 3, 3) => 50
      | (0, 3, 5) => 50
      | (0, 3, 6) => 50
      | (0, 5, 3) => 50
      | (0, 5, 5) => 50
      | (0, 5, 6) => 50
      | (0, 6, 3) => 50
      | (0, 6, 5) => 50
      | (0, 6, 6) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_1, returnGen 1);
        (100-_weight_1, returnGen 0)
      ]) (fun n1 =>
      (let _weight_2 := match (size, stack1, stack2) with
      | (0, 3, 3) => 50
      | (0, 3, 5) => 50
      | (0, 3, 6) => 50
      | (0, 5, 3) => 50
      | (0, 5, 5) => 50
      | (0, 5, 6) => 50
      | (0, 6, 3) => 50
      | (0, 6, 5) => 50
      | (0, 6, 6) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_2, returnGen 2);
        (100-_weight_2, returnGen 0)
      ]) (fun n2 =>
      (let _weight_4 := match (size, stack1, stack2) with
      | (0, 3, 3) => 50
      | (0, 3, 5) => 50
      | (0, 3, 6) => 50
      | (0, 5, 3) => 50
      | (0, 5, 5) => 50
      | (0, 5, 6) => 50
      | (0, 6, 3) => 50
      | (0, 6, 5) => 50
      | (0, 6, 6) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_4, returnGen 4);
        (100-_weight_4, returnGen 0)
      ]) (fun n4 =>
        returnGen (n1 + n2 + n4)
      )))))) 
      (fun p1 => 
        (returnGen (Var p1))))); 
      (* Bool *) (match (size, stack1, stack2) with
      | (0, 3, 3) => 5
      | (0, 3, 5) => 55
      | (0, 3, 6) => 61
      | (0, 5, 3) => 5
      | (0, 5, 5) => 67
      | (0, 5, 6) => 53
      | (0, 6, 3) => 5
      | (0, 6, 5) => 49
      | (0, 6, 6) => 72
      | _ => 500
      end,
      (bindGen 
      (* GenBool1 *) (let _weight_true := match (size, stack1, stack2) with
      | (0, 3, 3) => 50
      | (0, 3, 5) => 50
      | (0, 3, 6) => 50
      | (0, 5, 3) => 50
      | (0, 5, 5) => 50
      | (0, 5, 6) => 50
      | (0, 6, 3) => 50
      | (0, 6, 5) => 50
      | (0, 6, 6) => 50
      | _ => 500
      end
      in
      freq [
        (_weight_true, returnGen true);
        (100-_weight_true, returnGen false)
      ]) 
      (fun p1 => 
        (returnGen (Bool p1)))))])
  | S size1 => 
    (* Frequency4 *) (freq [
      (* Var *) (match (size, stack1, stack2) with
      | (1, 3, 3) => 88
      | (1, 3, 5) => 34
      | (1, 3, 6) => 58
      | (1, 5, 3) => 74
      | (1, 5, 5) => 47
      | (1, 5, 6) => 54
      | (1, 6, 3) => 92
      | (1, 6, 5) => 44
      | (1, 6, 6) => 56
      | (2, 3, 3) => 14
      | (2, 3, 5) => 8
      | (2, 3, 6) => 8
      | (2, 5, 3) => 21
      | (2, 5, 5) => 9
      | (2, 5, 6) => 7
      | (2, 6, 3) => 8
      | (2, 6, 5) => 13
      | (2, 6, 6) => 7
      | (3, 3, 3) => 5
      | (3, 3, 5) => 5
      | (3, 3, 6) => 5
      | (3, 5, 3) => 6
      | (3, 5, 5) => 5
      | (3, 5, 6) => 5
      | (3, 6, 3) => 5
      | (3, 6, 5) => 5
      | (3, 6, 6) => 5
      | (4, 0, 3) => 5
      | (4, 0, 5) => 5
      | (4, 0, 6) => 5
      | (5, 0, 0) => 5
      | _ => 500
      end,
      (bindGen 
      (* GenNat2 *)
      (let _weight_1 := match (size, stack1, stack2) with
      | (1, 3, 3) => 50
      | (1, 3, 5) => 50
      | (1, 3, 6) => 50
      | (1, 5, 3) => 50
      | (1, 5, 5) => 50
      | (1, 5, 6) => 50
      | (1, 6, 3) => 50
      | (1, 6, 5) => 50
      | (1, 6, 6) => 50
      | (2, 3, 3) => 50
      | (2, 3, 5) => 50
      | (2, 3, 6) => 50
      | (2, 5, 3) => 50
      | (2, 5, 5) => 50
      | (2, 5, 6) => 50
      | (2, 6, 3) => 50
      | (2, 6, 5) => 50
      | (2, 6, 6) => 50
      | (3, 3, 3) => 50
      | (3, 3, 5) => 50
      | (3, 3, 6) => 50
      | (3, 5, 3) => 50
      | (3, 5, 5) => 50
      | (3, 5, 6) => 50
      | (3, 6, 3) => 50
      | (3, 6, 5) => 50
      | (3, 6, 6) => 50
      | (4, 0, 3) => 50
      | (4, 0, 5) => 50
      | (4, 0, 6) => 50
      | (5, 0, 0) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_1, returnGen 1);
        (100-_weight_1, returnGen 0)
      ]) (fun n1 =>
      (let _weight_2 := match (size, stack1, stack2) with
      | (1, 3, 3) => 50
      | (1, 3, 5) => 50
      | (1, 3, 6) => 50
      | (1, 5, 3) => 50
      | (1, 5, 5) => 50
      | (1, 5, 6) => 50
      | (1, 6, 3) => 50
      | (1, 6, 5) => 50
      | (1, 6, 6) => 50
      | (2, 3, 3) => 50
      | (2, 3, 5) => 50
      | (2, 3, 6) => 50
      | (2, 5, 3) => 50
      | (2, 5, 5) => 50
      | (2, 5, 6) => 50
      | (2, 6, 3) => 50
      | (2, 6, 5) => 50
      | (2, 6, 6) => 50
      | (3, 3, 3) => 50
      | (3, 3, 5) => 50
      | (3, 3, 6) => 50
      | (3, 5, 3) => 50
      | (3, 5, 5) => 50
      | (3, 5, 6) => 50
      | (3, 6, 3) => 50
      | (3, 6, 5) => 50
      | (3, 6, 6) => 50
      | (4, 0, 3) => 50
      | (4, 0, 5) => 50
      | (4, 0, 6) => 50
      | (5, 0, 0) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_2, returnGen 2);
        (100-_weight_2, returnGen 0)
      ]) (fun n2 =>
      (let _weight_4 := match (size, stack1, stack2) with
      | (1, 3, 3) => 50
      | (1, 3, 5) => 50
      | (1, 3, 6) => 50
      | (1, 5, 3) => 50
      | (1, 5, 5) => 50
      | (1, 5, 6) => 50
      | (1, 6, 3) => 50
      | (1, 6, 5) => 50
      | (1, 6, 6) => 50
      | (2, 3, 3) => 50
      | (2, 3, 5) => 50
      | (2, 3, 6) => 50
      | (2, 5, 3) => 50
      | (2, 5, 5) => 50
      | (2, 5, 6) => 50
      | (2, 6, 3) => 50
      | (2, 6, 5) => 50
      | (2, 6, 6) => 50
      | (3, 3, 3) => 50
      | (3, 3, 5) => 50
      | (3, 3, 6) => 50
      | (3, 5, 3) => 50
      | (3, 5, 5) => 50
      | (3, 5, 6) => 50
      | (3, 6, 3) => 50
      | (3, 6, 5) => 50
      | (3, 6, 6) => 50
      | (4, 0, 3) => 50
      | (4, 0, 5) => 50
      | (4, 0, 6) => 50
      | (5, 0, 0) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_4, returnGen 4);
        (100-_weight_4, returnGen 0)
      ]) (fun n4 =>
        returnGen (n1 + n2 + n4)
      )))))) 
      (fun p1 => 
        (returnGen (Var p1))))); 
      (* Bool *) (match (size, stack1, stack2) with
      | (1, 3, 3) => 5
      | (1, 3, 5) => 52
      | (1, 3, 6) => 28
      | (1, 5, 3) => 5
      | (1, 5, 5) => 42
      | (1, 5, 6) => 31
      | (1, 6, 3) => 5
      | (1, 6, 5) => 52
      | (1, 6, 6) => 38
      | (2, 3, 3) => 5
      | (2, 3, 5) => 11
      | (2, 3, 6) => 6
      | (2, 5, 3) => 5
      | (2, 5, 5) => 7
      | (2, 5, 6) => 5
      | (2, 6, 3) => 5
      | (2, 6, 5) => 15
      | (2, 6, 6) => 8
      | (3, 3, 3) => 5
      | (3, 3, 5) => 5
      | (3, 3, 6) => 64
      | (3, 5, 3) => 5
      | (3, 5, 5) => 5
      | (3, 5, 6) => 5
      | (3, 6, 3) => 5
      | (3, 6, 5) => 5
      | (3, 6, 6) => 64
      | (4, 0, 3) => 5
      | (4, 0, 5) => 5
      | (4, 0, 6) => 5
      | (5, 0, 0) => 5
      | _ => 500
      end,
      (bindGen 
      (* GenBool2 *) (let _weight_true := match (size, stack1, stack2) with
      | (1, 3, 3) => 50
      | (1, 3, 5) => 50
      | (1, 3, 6) => 50
      | (1, 5, 3) => 50
      | (1, 5, 5) => 50
      | (1, 5, 6) => 50
      | (1, 6, 3) => 50
      | (1, 6, 5) => 50
      | (1, 6, 6) => 50
      | (2, 3, 3) => 50
      | (2, 3, 5) => 50
      | (2, 3, 6) => 50
      | (2, 5, 3) => 50
      | (2, 5, 5) => 50
      | (2, 5, 6) => 50
      | (2, 6, 3) => 50
      | (2, 6, 5) => 50
      | (2, 6, 6) => 50
      | (3, 3, 3) => 50
      | (3, 3, 5) => 50
      | (3, 3, 6) => 50
      | (3, 5, 3) => 50
      | (3, 5, 5) => 50
      | (3, 5, 6) => 50
      | (3, 6, 3) => 50
      | (3, 6, 5) => 50
      | (3, 6, 6) => 50
      | (4, 0, 3) => 50
      | (4, 0, 5) => 50
      | (4, 0, 6) => 50
      | (5, 0, 0) => 50
      | _ => 500
      end
      in
      freq [
        (_weight_true, returnGen true);
        (100-_weight_true, returnGen false)
      ]) 
      (fun p1 => 
        (returnGen (Bool p1))))); 
      (* Abs *) (match (size, stack1, stack2) with
      | (1, 3, 3) => 95
      | (1, 3, 5) => 80
      | (1, 3, 6) => 84
      | (1, 5, 3) => 93
      | (1, 5, 5) => 75
      | (1, 5, 6) => 66
      | (1, 6, 3) => 95
      | (1, 6, 5) => 83
      | (1, 6, 6) => 86
      | (2, 3, 3) => 72
      | (2, 3, 5) => 44
      | (2, 3, 6) => 23
      | (2, 5, 3) => 71
      | (2, 5, 5) => 44
      | (2, 5, 6) => 31
      | (2, 6, 3) => 81
      | (2, 6, 5) => 35
      | (2, 6, 6) => 40
      | (3, 3, 3) => 94
      | (3, 3, 5) => 15
      | (3, 3, 6) => 87
      | (3, 5, 3) => 15
      | (3, 5, 5) => 12
      | (3, 5, 6) => 9
      | (3, 6, 3) => 92
      | (3, 6, 5) => 5
      | (3, 6, 6) => 85
      | (4, 0, 3) => 95
      | (4, 0, 5) => 6
      | (4, 0, 6) => 95
      | (5, 0, 0) => 95
      | _ => 500
      end,
      (bindGen (genTyp 2 stack2 2) 
      (fun p1 => 
        (bindGen (genExpr size1 stack2 5) 
        (fun p2 => 
          (returnGen (Abs p1 p2))))))); 
      (* App *) (match (size, stack1, stack2) with
      | (1, 3, 3) => 88
      | (1, 3, 5) => 65
      | (1, 3, 6) => 60
      | (1, 5, 3) => 68
      | (1, 5, 5) => 28
      | (1, 5, 6) => 58
      | (1, 6, 3) => 90
      | (1, 6, 5) => 20
      | (1, 6, 6) => 59
      | (2, 3, 3) => 95
      | (2, 3, 5) => 87
      | (2, 3, 6) => 92
      | (2, 5, 3) => 95
      | (2, 5, 5) => 87
      | (2, 5, 6) => 92
      | (2, 6, 3) => 95
      | (2, 6, 5) => 86
      | (2, 6, 6) => 92
      | (3, 3, 3) => 5
      | (3, 3, 5) => 95
      | (3, 3, 6) => 5
      | (3, 5, 3) => 95
      | (3, 5, 5) => 91
      | (3, 5, 6) => 95
      | (3, 6, 3) => 5
      | (3, 6, 5) => 95
      | (3, 6, 6) => 7
      | (4, 0, 3) => 26
      | (4, 0, 5) => 95
      | (4, 0, 6) => 6
      | (5, 0, 0) => 95
      | _ => 500
      end,
      (bindGen (genExpr size1 stack2 3) 
      (fun p1 => 
        (bindGen (genExpr size1 stack2 6) 
        (fun p2 => 
          (returnGen (App p1 p2)))))))])
  end.

Definition gSized :=
  (genExpr 5 0 0).

Definition test_prop_SinglePreserve :=
forAll gSized (fun (e: Expr) =>
  prop_SinglePreserve e).

(*! QuickChick test_prop_SinglePreserve. *)

Definition test_prop_MultiPreserve :=
forAll gSized (fun (e: Expr) =>
  prop_MultiPreserve e).

(*! QuickChick test_prop_MultiPreserve. *)
          