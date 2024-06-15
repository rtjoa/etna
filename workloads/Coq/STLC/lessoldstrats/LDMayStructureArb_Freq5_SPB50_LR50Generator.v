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
      | (0, 3, 3) => 91
      | (0, 3, 5) => 93
      | (0, 3, 6) => 96
      | (0, 5, 3) => 88
      | (0, 5, 5) => 1
      | (0, 5, 6) => 90
      | (0, 6, 3) => 90
      | (0, 6, 5) => 93
      | (0, 6, 6) => 86
      | _ => 500
      end,
      (bindGen 
      arbitrary 
      (fun p1 => 
        (returnGen (Var p1))))); 
      (* Bool *) (match (size, stack1, stack2) with
      | (0, 3, 3) => 0
      | (0, 3, 5) => 91
      | (0, 3, 6) => 41
      | (0, 5, 3) => 0
      | (0, 5, 5) => 89
      | (0, 5, 6) => 13
      | (0, 6, 3) => 0
      | (0, 6, 5) => 87
      | (0, 6, 6) => 92
      | _ => 500
      end,
      (bindGen 
      arbitrary 
      (fun p1 => 
        (returnGen (Bool p1)))))])
  | S size1 => 
    (* Frequency4 *) (freq [
      (* Var *) (match (size, stack1, stack2) with
      | (1, 3, 3) => 2
      | (1, 3, 5) => 1
      | (1, 3, 6) => 2
      | (1, 5, 3) => 88
      | (1, 5, 5) => 2
      | (1, 5, 6) => 62
      | (1, 6, 3) => 7
      | (1, 6, 5) => 7
      | (1, 6, 6) => 40
      | (2, 3, 3) => 1
      | (2, 3, 5) => 3
      | (2, 3, 6) => 0
      | (2, 5, 3) => 52
      | (2, 5, 5) => 36
      | (2, 5, 6) => 58
      | (2, 6, 3) => 0
      | (2, 6, 5) => 2
      | (2, 6, 6) => 0
      | (3, 3, 3) => 30
      | (3, 3, 5) => 20
      | (3, 3, 6) => 31
      | (3, 5, 3) => 0
      | (3, 5, 5) => 20
      | (3, 5, 6) => 0
      | (3, 6, 3) => 30
      | (3, 6, 5) => 36
      | (3, 6, 6) => 31
      | (4, 0, 3) => 4
      | (4, 0, 5) => 0
      | (4, 0, 6) => 6
      | (5, 0, 0) => 0
      | _ => 500
      end,
      (bindGen 
      arbitrary 
      (fun p1 => 
        (returnGen (Var p1))))); 
      (* Bool *) (match (size, stack1, stack2) with
      | (1, 3, 3) => 0
      | (1, 3, 5) => 20
      | (1, 3, 6) => 84
      | (1, 5, 3) => 1
      | (1, 5, 5) => 12
      | (1, 5, 6) => 8
      | (1, 6, 3) => 0
      | (1, 6, 5) => 3
      | (1, 6, 6) => 22
      | (2, 3, 3) => 0
      | (2, 3, 5) => 3
      | (2, 3, 6) => 0
      | (2, 5, 3) => 7
      | (2, 5, 5) => 37
      | (2, 5, 6) => 31
      | (2, 6, 3) => 0
      | (2, 6, 5) => 75
      | (2, 6, 6) => 0
      | (3, 3, 3) => 30
      | (3, 3, 5) => 21
      | (3, 3, 6) => 53
      | (3, 5, 3) => 0
      | (3, 5, 5) => 18
      | (3, 5, 6) => 0
      | (3, 6, 3) => 30
      | (3, 6, 5) => 76
      | (3, 6, 6) => 61
      | (4, 0, 3) => 4
      | (4, 0, 5) => 0
      | (4, 0, 6) => 66
      | (5, 0, 0) => 0
      | _ => 500
      end,
      (bindGen 
      arbitrary 
      (fun p1 => 
        (returnGen (Bool p1))))); 
      (* Abs *) (match (size, stack1, stack2) with
      | (1, 3, 3) => 90
      | (1, 3, 5) => 89
      | (1, 3, 6) => 96
      | (1, 5, 3) => 20
      | (1, 5, 5) => 83
      | (1, 5, 6) => 87
      | (1, 6, 3) => 94
      | (1, 6, 5) => 93
      | (1, 6, 6) => 97
      | (2, 3, 3) => 0
      | (2, 3, 5) => 66
      | (2, 3, 6) => 0
      | (2, 5, 3) => 79
      | (2, 5, 5) => 76
      | (2, 5, 6) => 52
      | (2, 6, 3) => 21
      | (2, 6, 5) => 81
      | (2, 6, 6) => 16
      | (3, 3, 3) => 80
      | (3, 3, 5) => 76
      | (3, 3, 6) => 72
      | (3, 5, 3) => 0
      | (3, 5, 5) => 60
      | (3, 5, 6) => 0
      | (3, 6, 3) => 80
      | (3, 6, 5) => 34
      | (3, 6, 6) => 55
      | (4, 0, 3) => 90
      | (4, 0, 5) => 0
      | (4, 0, 6) => 81
      | (5, 0, 0) => 97
      | _ => 500
      end,
      (bindGen (genTyp 2 stack2 2) 
      (fun p1 => 
        (bindGen (genExpr size1 stack2 5) 
        (fun p2 => 
          (returnGen (Abs p1 p2))))))); 
      (* App *) (match (size, stack1, stack2) with
      | (1, 3, 3) => 97
      | (1, 3, 5) => 94
      | (1, 3, 6) => 95
      | (1, 5, 3) => 66
      | (1, 5, 5) => 78
      | (1, 5, 6) => 11
      | (1, 6, 3) => 97
      | (1, 6, 5) => 73
      | (1, 6, 6) => 95
      | (2, 3, 3) => 97
      | (2, 3, 5) => 89
      | (2, 3, 6) => 96
      | (2, 5, 3) => 44
      | (2, 5, 5) => 36
      | (2, 5, 6) => 60
      | (2, 6, 3) => 97
      | (2, 6, 5) => 46
      | (2, 6, 6) => 96
      | (3, 3, 3) => 30
      | (3, 3, 5) => 62
      | (3, 3, 6) => 31
      | (3, 5, 3) => 97
      | (3, 5, 5) => 78
      | (3, 5, 6) => 96
      | (3, 6, 3) => 30
      | (3, 6, 5) => 36
      | (3, 6, 6) => 50
      | (4, 0, 3) => 23
      | (4, 0, 5) => 96
      | (4, 0, 6) => 20
      | (5, 0, 0) => 0
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
          
